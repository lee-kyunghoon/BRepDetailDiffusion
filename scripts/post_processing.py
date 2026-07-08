from glob import glob
import time

import numpy as np
import os
import math
import torch

from OCC.Extend.DataExchange import write_step_file
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_IN
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shell, topods
from OCC.Core.ShapeFix import ShapeFix_Solid, ShapeFix_Shape, ShapeFix_Face, ShapeFix_Shell
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeFace
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_GlueEnum
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Geom import Geom_CylindricalSurface

from brepdiff.primitives.uvgrid import UvGrid
from brepdiff.postprocessing.postprocessor import Postprocessor
from brepdiff.utils.common import timeout_wrapper
from multiprocessing import Process, Queue

from try_fit_cylinder import try_fit_cylinder_params_from_points

OCC_THRESHOLD = 0.5
PROCESS_TIMEOUT_SEC = 30

def _time_exceeded(start_time, timeout_sec):
    return (time.monotonic() - start_time) > timeout_sec

@timeout_wrapper
def try_combined_bspline_occ(uvgrid, verbose=True):
    """
    Postprocessor의 PSR occupancy + BOPAlgo_CellsBuilder 결합 방식.
    """

    start_time = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    # ─── Phase 1: Postprocessor occupancy pipeline ───
    uvgrid.to_tensor(device)
    uvgrid.upscale_grid(new_grid_size=14)
    pp = Postprocessor(
        uvgrid=uvgrid,
        psr_uvgrid_res=32,
        smooth_extension=False,
        grid_res=256,
        device=device,
    )

    if verbose:
        print("Phase 1: PSR occupancy...")
    try:
        occupancy = pp.get_occupancy()
    except Exception as e:
        if verbose:
            print(f"  PSR occupancy 실패: {e}")
        return None

    if verbose:
        print("Phase 1: Partition grid (winding numbers)...")
    try:
        partition, n_partition = pp.get_extended_partition_grid()
    except Exception as e:
        if verbose:
            print(f"  Partition 계산 실패: {e}")
        return None

    if verbose:
        print("Phase 1: Vote occupancy...")
    _, partition_occ, _ = pp.vote_occupancy(occupancy, partition, n_partition)

    grid_res = pp.grid_res
    grid_min = pp.grid_min_range
    grid_max = pp.grid_max_range
    grid_min_t = torch.as_tensor(grid_min, dtype=torch.float32, device=device)
    grid_max_t = torch.as_tensor(grid_max, dtype=torch.float32, device=device)
    occ_t = partition_occ.to(device=device)
    
    if verbose:
        occ_count = int(partition_occ.sum().item())
        occ_total = int(partition_occ.numel())
        occ_frac = occ_count / occ_total if occ_total > 0 else 0.0
        print(f"  Occupied: {occ_count}/{occ_total} ({occ_frac:.1%})")

    # ─── Phase 2: B-spline surface fitting (원래 좌표계) ───
    coord_ext_scaled = pp.scaled_uvgrid_extend.coord.cpu().numpy()
    empty_mask = pp.scaled_uvgrid.empty_mask.cpu().numpy()
    n_non_empty = int((~empty_mask).sum())

    axis_scale = pp.axis_scale.cpu().numpy()
    axis_offset = pp.axis_offset.cpu().numpy()
    axis_scale_t = pp.axis_scale.to(device=device, dtype=torch.float32)
    axis_offset_t = pp.axis_offset.to(device=device, dtype=torch.float32)

    # scaled → original
    coord_origin = pp.scaled_uvgrid.coord.cpu().numpy()
    coord_origin = (coord_origin.reshape(-1, 3) - axis_offset) / axis_scale
    coord_origin = coord_origin.reshape(pp.scaled_uvgrid.coord.shape)

    coord_ext = (coord_ext_scaled.reshape(-1, 3) - axis_offset) / axis_scale
    coord_ext = coord_ext.reshape(coord_ext_scaled.shape)
    #coord_ext = coord_ext_scaled

    if verbose:
        print(f"Phase 2: B-spline fitting ({n_non_empty} faces, extended grid)...")

    occ_faces = []
    cyl_candidates = []  # (face_idx, pts_flat, pts_grid, H, W, center, axis, radius)
    leftover1 = []
    for i in range(len(empty_mask)):
        if _time_exceeded(start_time, PROCESS_TIMEOUT_SEC):
            if verbose:
                print(f"  timeout exceeded during face fitting ({PROCESS_TIMEOUT_SEC}s), skip")
            return None
        
        if empty_mask[i]:
            continue
        pts = coord_ext[i]
        origin_pts = coord_origin[i]
        H, W = pts.shape[:2]

        cyl_params = try_fit_cylinder_params_from_points(pts.reshape(-1, 3))

        if cyl_params is not None:
            origin_cyl_params = try_fit_cylinder_params_from_points(origin_pts.reshape(-1, 3))
            if origin_cyl_params is None:
                leftover1.append((i, None, pts, H, W, None, None, None))
                continue

            original_center, original_axis, original_radius, _ = origin_cyl_params
            _, _, _, flat_pts = cyl_params
            cyl_candidates.append((i, flat_pts, pts, H, W, original_center, original_axis, original_radius))
            if verbose:
                print(f"  Face {i}: cylinder 후보 (r={original_radius:.4f})")
            continue

        arr = TColgp_Array2OfPnt(1, H, 1, W)
        if verbose:
            print(f"  Face {i}: bspline ")
        for u in range(H):
            for v in range(W):
                x, y, z = pts[u, v]
                arr.SetValue(u + 1, v + 1, gp_Pnt(float(x), float(y), float(z)))
        try:
            approx = GeomAPI_PointsToBSplineSurface(arr, 1, 3, GeomAbs_C2, 1e-2)
            surf = approx.Surface()
            face = BRepBuilderAPI_MakeFace(surf, 1e-3).Face()
            occ_faces.append(face)
        except Exception as e:
            if verbose:
                print(f"  Face {i}: fitting 실패 - {e}")

    # Cylinder 후보 처리:
    if cyl_candidates:
        cyl_faces, leftover = _group_and_make_cylinder_faces(cyl_candidates, verbose=verbose)
        occ_faces.extend(cyl_faces)
        leftover1.extend(leftover)
        for _, _, pts_grid, H, W, _, _, _ in leftover1:
            arr = TColgp_Array2OfPnt(1, H, 1, W)
            for u in range(H):
                for v in range(W):
                    x, y, z = pts_grid[u, v]
                    arr.SetValue(u + 1, v + 1, gp_Pnt(float(x), float(y), float(z)))
            try:
                approx = GeomAPI_PointsToBSplineSurface(arr, 1, 3, GeomAbs_C2, 1e-2)
                face = BRepBuilderAPI_MakeFace(approx.Surface(), 1e-3).Face()
                occ_faces.append(face)
            except Exception:
                pass

    if verbose:
        print(f"  {len(occ_faces)}/{n_non_empty} surfaces fitted")

    if len(occ_faces) < 2:
        if verbose:
            print("  면이 부족")
        return None

    # ─── Phase 3: CellsBuilder → cell 분할 → MakeContainers ───
    if verbose:
        print("Phase 3: BOPAlgo_CellsBuilder...")

    solids = []
    tol = 1e-3
    sewing = BRepBuilderAPI_Sewing(tol)
    sewing.SetTolerance(tol)
    for face in occ_faces:
        sewing.Add(face)
    sewing.Perform()
    shape = sewing.SewedShape()
    write_step_file(shape, "shell.step")

    if not solids:
        if verbose:
            print("  MakerVolume fallback...")
        
        shape_list = TopTools_ListOfShape()
        for face in occ_faces:
            shape_list.Append(face)

        maker = BOPAlgo_MakerVolume()
        maker.SetArguments(shape_list)
        maker.SetNonDestructive(True)
        maker.SetRunParallel(False)
        maker.SetAvoidInternalShapes(True)
        maker.SetToFillHistory(True)
        maker.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueOff)
        maker.SetFuzzyValue(1e-5)
        maker.SetIntersect(True)
        maker.Perform()

        if maker.HasErrors():
            if verbose:
                print("  MakerVolume 실패")
            return None

        result_shape = maker.Shape()
        write_step_file(result_shape, "volume_result.step")
        exp = TopExp_Explorer(result_shape, TopAbs_SOLID)
        while exp.More():
            solids.append(topods.Solid(exp.Current()))
            exp.Next()

        if verbose:
            print(f"  MakerVolume → {len(solids)} solids")

    # ─── Phase 4: Occupancy 기반 cell 선택 + 병합 ───
    if verbose:
        print(f"Phase 4: Occupancy scoring ({len(solids)} cells)...")

    def _occ_query(pts):
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
        pts_scaled = pts_t * axis_scale_t + axis_offset_t
        denom = (grid_max_t - grid_min_t).clamp_min(1e-8)
        grid_coords = (pts_scaled - grid_min_t) / denom * (grid_res - 1)
        idx = torch.round(grid_coords).to(torch.long)
        idx = idx.clamp(0, grid_res - 1)
        occ_values = occ_t[idx[:, 0], idx[:, 1], idx[:, 2]]
        return float(occ_values.float().mean().item())

    scored_solids = []
    rng = np.random.RandomState(42)

    for si, solid in enumerate(solids):
        if _time_exceeded(start_time, PROCESS_TIMEOUT_SEC):
            if verbose:
                print(f"  timeout exceeded during cell scoring ({PROCESS_TIMEOUT_SEC}s), skip")
            return None

        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        volume = abs(props.Mass())

        if volume < 1e-10:
            if verbose:
                print(f"  Cell {si}: vol={volume:.2e} → degenerate solid, skip")
            continue

        bbox = Bnd_Box()
        brepbndlib.Add(solid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if dx < 1e-10 or dy < 1e-10 or dz < 1e-10:
            if verbose:
                print(f"  Cell {si}: degenerate bbox "
                      f"({dx:.2e}, {dy:.2e}, {dz:.2e}) → skip")
            continue

        n_sample = 2000
        sample_pts = np.column_stack([
            rng.uniform(xmin, xmax, n_sample),
            rng.uniform(ymin, ymax, n_sample),
            rng.uniform(zmin, zmax, n_sample),
        ])

        classifier = BRepClass3d_SolidClassifier(solid)
        inside_pts = []
        for pt in sample_pts:
            classifier.Perform(gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), 1e-4)
            if classifier.State() == TopAbs_IN:
                inside_pts.append(pt)

        if not inside_pts:
            if verbose:
                print(f"  Cell {si}: vol={volume:.6f}, inside=0 → skip")
            continue

        inside_pts = np.array(inside_pts)
        occ_match = _occ_query(inside_pts)

        n_faces = 0
        exp2 = TopExp_Explorer(solid, TopAbs_FACE)
        while exp2.More():
            n_faces += 1
            exp2.Next()

        if verbose:
            print(f"  Cell {si}: vol={volume:.6f}, faces={n_faces}, "
                  f"inside={len(inside_pts)}, occ_match={occ_match:.1%}")

        scored_solids.append((solid, occ_match, volume, n_faces))

    if not scored_solids:
        if verbose:
            print("  유효한 cell 없음")
        return None

    # Occupancy match threshold로 필터링
    occ_threshold = OCC_THRESHOLD
    good_solids = [(s, m, v, f) for s, m, v, f in scored_solids if m > occ_threshold]

    if not good_solids:
        # threshold 이상이 없으면 best 1개
        best = max(scored_solids, key=lambda x: x[1])
        good_solids = [best]

    # 최대 cell 대비 너무 작은 cell 제외 (Fuse 시 복잡한 topology 방지)
    if len(good_solids) > 1:
        max_vol = max(v for _, _, v, _ in good_solids)
        min_vol_ratio = 0.01
        before_count = len(good_solids)
        good_solids = [(s, m, v, f) for s, m, v, f in good_solids
                       if v >= min_vol_ratio * max_vol]
        if not good_solids:
            # 모두 제외되면 가장 큰 것 유지
            good_solids = [max(scored_solids, key=lambda x: x[2])]
        if verbose and len(good_solids) < before_count:
            print(f"  Volume 필터: {before_count} → {len(good_solids)} cells "
                  f"(min_vol={min_vol_ratio*max_vol:.6f})")

    if verbose:
        total_faces = sum(f for _, _, _, f in good_solids)
        total_vol = sum(v for _, _, v, _ in good_solids)
        print(f"  선택: {len(good_solids)}/{len(scored_solids)} cells "
              f"(threshold={occ_threshold}), total_faces={total_faces}, total_vol={total_vol:.6f}")

    # 선택된 solids를 BRepAlgoAPI_Fuse로 병합
    if len(good_solids) == 1:
        merged = good_solids[0][0]
    else:
        if verbose:
            print("  Fusing selected cells...")
        start_time = time.monotonic()
        merged = good_solids[0][0]
        for i, (s, m, v, f) in enumerate(good_solids[1:], 1):
            if _time_exceeded(start_time, PROCESS_TIMEOUT_SEC):
                if verbose:
                    print(f"  timeout exceeded during fuse ({PROCESS_TIMEOUT_SEC}s), skip")
                return None
            fuse_op = BRepAlgoAPI_Fuse(merged, s)
            if fuse_op.IsDone():
                merged = fuse_op.Shape()
                if verbose:
                    print(f"    Fuse {i}/{len(good_solids)-1} done")
            else:
                if verbose:
                    print(f"    Fuse {i} 실패, skip")

    result_solid = None
    if merged.ShapeType() == TopAbs_SOLID:
        result_solid = merged
    else:
        write_step_file(merged, 'Lpart2_12_commpound.step')
        exp = TopExp_Explorer(merged, TopAbs_SOLID)
        if exp.More():
            result_solid = topods.Solid(exp.Current())

    if result_solid is None:
        if verbose:
            print("  병합 후 solid 추출 실패")
        return None
    
    n_faces = 0
    exp2 = TopExp_Explorer(result_solid, TopAbs_FACE)
    while exp2.More():
        n_faces += 1
        exp2.Next()
    if verbose:
        print(f"  최종 결과: {n_faces} faces")

    if n_faces < 5:
        return None

    return result_solid

def _weighted_median(values, weights):
    """
    가중 중앙값(weighted median)을 계산한다.
    누적 가중치가 전체의 50%를 처음 넘는 값을 반환한다.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        finite_v = v[np.isfinite(v)]
        if finite_v.size == 0:
            return None
        return float(np.median(finite_v))

    v = v[valid]
    w = w[valid]

    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]

    csum = np.cumsum(w_sorted)
    cutoff = 0.5 * csum[-1]
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])

def _compute_circular_angle_range(angles):
    """
    [-π, π] 범위의 각도 배열에서 wrap-around를 고려한 호의 범위를 계산.

    정렬된 각도 사이의 최대 gap을 찾아 호의 시작/끝을 결정한다.
    호는 [start, end] 연속 구간으로 표현되며 end >= start (필요 시 +2π).

    Returns:
        (span, start, end):
            - span: 호의 크기 (rad)
            - start, end: 호의 시작/끝 (end >= start, end - start = span)
    """
    if len(angles) < 2:
        return 0.0, 0.0, 0.0

    angles_sorted = np.sort(angles)
    gaps = np.diff(angles_sorted)
    wrap_gap = (2.0 * np.pi) - angles_sorted[-1] + angles_sorted[0]
    gaps_all = np.append(gaps, wrap_gap)

    max_gap_idx = int(np.argmax(gaps_all))
    max_gap = gaps_all[max_gap_idx]
    span = 2.0 * np.pi - max_gap

    if max_gap_idx == len(gaps_all) - 1:
        # wrap-around gap이 최대 → 호는 angles_sorted[0] ~ angles_sorted[-1]
        start = float(angles_sorted[0])
        end = float(angles_sorted[-1])
    else:
        # 호는 max_gap 직후 ~ max_gap 직전
        start = float(angles_sorted[max_gap_idx + 1])
        end = float(angles_sorted[max_gap_idx])
        if end < start:
            end += 2.0 * np.pi

    return float(span), start, end


def _group_and_make_cylinder_faces(cyl_candidates, verbose=False):

    # ================================================================
    # 1) 그룹화: axis 유사 + 반지름 유사 + perpendicular distance
    # ================================================================
    used = [False] * len(cyl_candidates)
    groups = []

    for i in range(len(cyl_candidates)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        _, _, _, _, _, center_i, axis_i, radius_i = cyl_candidates[i]

        for j in range(i + 1, len(cyl_candidates)):
            if used[j]:
                continue
            _, _, _, _, _, center_j, axis_j, radius_j = cyl_candidates[j]

            axis_dot = abs(np.dot(axis_i, axis_j))
            if axis_dot < 0.90:
                continue
            r_rel = abs(radius_i - radius_j) / max(radius_i, radius_j, 1e-8)
            if r_rel > 0.15:
                continue

            diff = center_j - center_i
            if abs(diff[0]) > 0.2 or abs(diff[1]) > 0.2 or abs(diff[2]) > 0.2:
                continue

            group.append(j)
            used[j] = True

        groups.append(group)

    if verbose:
        group_str = [[cyl_candidates[gi][0] for gi in g] for g in groups]
        print(f"  Cylinder 그룹: {group_str}")

    # ================================================================
    # 2) 그룹별 cylinder face 생성
    # ================================================================
    result_faces = []
    leftover = []

    for group in groups:
        if len(group) < 2:
            for gi in group:
                leftover.append(cyl_candidates[gi])
            if verbose:
                face_ids = [cyl_candidates[gi][0] for gi in group]
                print(f"  단독 cylinder 후보 → B-spline fallback: {face_ids}")
            continue

        # ── 축 방향 보정 후 평균 ──
        axes = np.array([cyl_candidates[gi][6] for gi in group])
        for k in range(1, len(axes)):
            if np.dot(axes[k], axes[0]) < 0:
                axes[k] = -axes[k]
        avg_axis = axes.mean(axis=0)
        avg_axis = avg_axis / np.linalg.norm(avg_axis)

        radii = np.array([cyl_candidates[gi][7] for gi in group])
        radius_weights = np.array(
            [max(float(len(cyl_candidates[gi][1])), 1.0) for gi in group],
            dtype=np.float64,
        )
        centers = np.array([cyl_candidates[gi][5] for gi in group])
        radius = _weighted_median(radii, radius_weights)
        if radius is None:
            for gi in group:
                leftover.append(cyl_candidates[gi])
            continue
        center_3d = centers.mean(axis=0)

        tmp = np.array([1, 0, 0]) if abs(avg_axis[0]) < 0.9 else np.array([0, 1, 0])
        e1 = np.cross(avg_axis, tmp)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(avg_axis, e1)

        all_pts = np.concatenate([cyl_candidates[gi][1] for gi in group], axis=0)
        pts_c = all_pts - center_3d
        axis_proj = pts_c @ avg_axis

        h_min = float(np.min(axis_proj)) - 0.01 * radius
        h_max = float(np.max(axis_proj)) + 0.01 * radius

        perp_all = pts_c - np.outer(axis_proj, avg_axis)
        coords_2d = np.column_stack([perp_all @ e1, perp_all @ e2])
        angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
        angle_span, angle_start, angle_end = _compute_circular_angle_range(angles)

        # full vs partial 판별: 90% 이상이면 full cylinder
        if angle_span > 0.9 * 2.0 * math.pi:
            u_min = 0.0
            u_max = 2.0 * math.pi
            arc_label = "full"
            is_full_cylinder = True
        else:
            is_full_cylinder = False
            # 부분 호: 양쪽에 5% 마진
            margin = 0.05 * angle_span
            u_min = angle_start - margin
            u_max = angle_end + margin

            # OCC 매개변수는 [0, 2π] 권장 → 음수 시작이면 시프트
            if u_min < 0:
                u_min += 2.0 * math.pi
                u_max += 2.0 * math.pi

            # 마진으로 인해 2π 초과하면 full로 처리
            if u_max - u_min >= 2.0 * math.pi:
                u_min = 0.0
                u_max = 2.0 * math.pi
                arc_label = "full (margin saturated)"
            else:
                arc_label = "partial"

        # ── OCC face 생성 ──
        ax3 = gp_Ax3(
            gp_Pnt(float(center_3d[0]), float(center_3d[1]), float(center_3d[2])),
            gp_Dir(float(avg_axis[0]), float(avg_axis[1]), float(avg_axis[2])),
        )
        cyl_surf = Geom_CylindricalSurface(ax3, float(radius))

        if is_full_cylinder:
            cyl_face = BRepBuilderAPI_MakeFace(
                cyl_surf, u_min, u_max, h_min, h_max, 1e-3
            ).Face()
            result_faces.append(cyl_face)
            if verbose:
                face_ids = [cyl_candidates[gi][0] for gi in group]
                print(f"  Cylinder face 생성: faces={face_ids}, "
                      f"r={radius:.4f}, "
                      f"angle=[{math.degrees(u_min):.1f}°, "
                      f"{math.degrees(u_max):.1f}°] ({arc_label}), "
                      f"h=[{h_min:.4f}, {h_max:.4f}]")
        else:
            for gi in group:
                leftover.append(cyl_candidates[gi])

    return result_faces, leftover



def _try_repair_face(face):
    """
    null triangulation인 face를 수리하여 유효한 face로 반환.
    
    수리 순서:
    1) ShapeFix_Face로 기하/위상 보정
    2) Surface + Wire로 face 재생성
    3) mesh 재시도
    
    모두 실패하면 None 반환.
    """
    topo_face = topods.Face(face)

    # ── 1단계: ShapeFix로 보정 시도 ──
    fixer = ShapeFix_Face(topo_face)
    fixer.SetPrecision(1e-3)
    fixer.Perform()
    fixed = fixer.Face()

    loc = TopLoc_Location()
    BRepMesh_IncrementalMesh(fixed, 0.01)
    tri = BRep_Tool.Triangulation(topods.Face(fixed), loc)
    if tri is not None:
        return fixed

    # ── 2단계: Surface + Wire로 재생성 ──
    surface = BRep_Tool.Surface(topo_face)
    if surface is None:
        return None

    # wire 추출
    wire_exp = TopExp_Explorer(topo_face, TopAbs_WIRE)
    if not wire_exp.More():
        # wire 없으면 surface만으로 face 생성
        try:
            maker = BRepBuilderAPI_MakeFace(surface, 1e-3)
            maker.Build()
            if maker.IsDone():
                new_face = maker.Face()
                BRepMesh_IncrementalMesh(new_face, 0.01)
                tri = BRep_Tool.Triangulation(new_face, loc)
                if tri is not None:
                    return new_face
        except Exception:
            pass
        return None

    # wire가 있으면 surface + wire로 재생성
    first_wire = topods.Wire(wire_exp.Current())
    try:
        maker = BRepBuilderAPI_MakeFace(surface, first_wire)
        wire_exp.Next()
        while wire_exp.More():
            maker.Add(topods.Wire(wire_exp.Current()))
            wire_exp.Next()
        maker.Build()
        if maker.IsDone():
            new_face = maker.Face()
            BRepMesh_IncrementalMesh(new_face, 0.01)
            tri = BRep_Tool.Triangulation(new_face, loc)
            if tri is not None:
                return new_face
    except Exception:
        pass

    # ── 3단계: deflection을 크게 해서 mesh 재시도 ──
    for deflection in [0.05, 0.1, 0.5]:
        BRepMesh_IncrementalMesh(topo_face, deflection)
        tri = BRep_Tool.Triangulation(topo_face, loc)
        if tri is not None:
            return topo_face

    return None


def _rebuild_shell_repairing_null_faces(shell, linear_deflection=0.01):
    """
    Shell 내부의 face를 순회하며 triangulation이 null인 face를
    수리하여 포함. 수리 불가능한 경우에만 제외.
    """
    BRepMesh_IncrementalMesh(shell, linear_deflection)

    builder = BRep_Builder()
    new_shell = TopoDS_Shell()
    builder.MakeShell(new_shell)

    exp = TopExp_Explorer(shell, TopAbs_FACE)
    n_total = 0
    n_repaired = 0
    n_skipped = 0

    while exp.More():
        face = exp.Current()
        n_total += 1
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(topods.Face(face), loc)

        if tri is not None:
            # 정상 face
            builder.Add(new_shell, face)
        else:
            # null → 수리 시도
            repaired = _try_repair_face(face)
            if repaired is not None:
                builder.Add(new_shell, repaired)
                n_repaired += 1
            else:
                n_skipped += 1
        exp.Next()

    if n_repaired > 0 or n_skipped > 0:
        print(f"  face 처리: {n_total}개 중 "
              f"{n_repaired}개 수리, {n_skipped}개 제거 불가")

    return new_shell


@timeout_wrapper
def heal_shape(shape, tolerance=1e-6, verbose=True):

    def _log(msg):
        if verbose:
            print(f"  [Heal] {msg}")

    # ── Step 1: Sewing ──────────────────────────────────────────────
    _log("Step 1: Sewing...")
    sewing = BRepBuilderAPI_Sewing(tolerance)
    sewing.Add(shape)
    sewing.Perform()
    sewn = sewing.SewedShape()
    if sewn.IsNull():
        _log("Sewing 결과 비어있음 — 원본 유지")
        sewn = shape
    else:
        _log("Sewing 완료")

    # ── Step 2: ShapeFix_Shape ───────────────────────────────────────
    _log("Step 2: ShapeFix_Shape...")
    fixer = ShapeFix_Shape(sewn)
    fixer.SetPrecision(tolerance)
    fixer.SetMinTolerance(tolerance * 0.1)
    fixer.SetMaxTolerance(tolerance * 10)
    fixer.FixSolidMode     = 1
    fixer.FixFreeShellMode = 1
    fixer.FixFreeFaceMode  = 1
    fixer.FixFreeWireMode  = 1
    fixer.Perform()
    fixed = fixer.Shape()
    if fixed.IsNull():
        _log("ShapeFix_Shape 결과 비어있음 — 이전 단계 결과 유지")
        fixed = sewn
    else:
        _log("ShapeFix_Shape 완료")

    # ── Step 3: Shell → Solid 재구성 (ShapeFix_Solid 대체) ──────────
    _log("Step 3: Shell → Solid 재구성...")
    solid_result = fixed

    # 이미 Solid가 있으면 그대로 사용
    exp_solid = TopExp_Explorer(fixed, TopAbs_SOLID)
    if exp_solid.More():
        _log("Solid 이미 존재 — Step 3 생략")

    else:
        # Shell을 수집해서 MakeSolid로 감싸기
        shells = []
        exp_shell = TopExp_Explorer(fixed, TopAbs_SHELL)
        while exp_shell.More():
            shells.append(topods.Shell(exp_shell.Current()))
            exp_shell.Next()

        if shells:
            _log(f"Shell {len(shells)}개 발견 — MakeSolid 시도...")
            maker = BRepBuilderAPI_MakeSolid()
            for shell in shells:
                # ShapeFix_Shell로 shell orientation 먼저 수정
                sf_shell = ShapeFix_Shell()
                sf_shell.SetPrecision(tolerance)
                sf_shell.FixOrientationMode = 1
                sf_shell.Init(shell)
                sf_shell.Perform()
                fixed_shell = topods.Shell(sf_shell.Shell())
                maker.Add(fixed_shell)

            if maker.IsDone():
                solid_result = maker.Solid()
                _log("MakeSolid 완료")
            else:
                _log("MakeSolid 실패 — 이전 단계 결과 유지")
        else:
            _log("Shell도 없음 — Step 3 생략")

    # ── Step 4: UnifySameDomain ─────────────────────────────────────
    _log("Step 4: UnifySameDomain...")
    unify = ShapeUpgrade_UnifySameDomain(solid_result, True, True, True)
    unify.Build()
    unified = unify.Shape()
    if unified.IsNull():
        _log("UnifySameDomain 결과 비어있음 — 이전 단계 결과 유지")
        unified = solid_result
    else:
        _log("UnifySameDomain 완료")

    # ── Step 5: UnifySameDomain 후 Solid 재확인 ─────────────────────
    # UnifySameDomain이 Solid → Shell로 되돌리는 경우가 있음
    if unified.ShapeType() != TopAbs_SOLID:
        _log(f"Shape type={unified.ShapeType()} — Solid 재구성 시도...")
        maker = BRepBuilderAPI_MakeSolid()
        exp = TopExp_Explorer(unified, TopAbs_SHELL)
        n = 0
        while exp.More():
            maker.Add(topods.Shell(exp.Current()))
            exp.Next()
            n += 1
        if n > 0 and maker.IsDone():
            unified = maker.Solid()
            _log("Solid 재구성 완료")
        else:
            _log("Solid 재구성 실패 — 이전 단계(MakeSolid) 결과 유지")
            unified = solid_result  # UnifySameDomain 이전 결과로 fallback

    # ── 최종 검증 ───────────────────────────────────────────────────
    analyzer = BRepCheck_Analyzer(unified)
    _log(f"최종 BRepCheck valid: {analyzer.IsValid()}")
    _log(f"Shape type: {unified.ShapeType()}")

    return unified


def extract_solid(shape):

    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(1e-3)
    fixer.Perform()

    fixed_shape = fixer.Shape()
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    
    exp_shell = TopExp_Explorer(fixed_shape, TopAbs_SHELL)
    while exp_shell.More():
        orig_shell = topods.Shell(exp_shell.Current())
        solid_fixer = ShapeFix_Solid()
        solid_fixer.SetPrecision(1e-3)
        solid = solid_fixer.SolidFromShell(orig_shell)
        analyzer = BRepCheck_Analyzer(solid)
        if analyzer.IsValid():
            builder.Add(compound, solid)
        else:
            fixer2 = ShapeFix_Shape(solid)
            fixer2.Perform()
            builder.Add(compound, fixer2.Shape())
        exp_shell.Next()

    return compound

def _count_wires_in_compound(compound):
    """TopoDS_Compound 안의 wire 개수 반환"""
    count = 0
    exp = TopExp_Explorer(compound, TopAbs_WIRE)
    while exp.More():
        count += 1
        exp.Next()
    return count

def check_watertight(shape):
    """
    BRep이 watertight solid인지 종합 검사.
    Returns: (is_watertight: bool, info: dict)
    """
    info = {}

    # 1. Shape Type 확인
    info["is_solid"] = shape.ShapeType() == TopAbs_SOLID

    # 2. Closed 플래그
    info["is_closed"] = shape.Closed()

    # 3. BRepCheck_Analyzer — 위상/기하 유효성
    analyzer = BRepCheck_Analyzer(shape)
    info["is_valid"] = analyzer.IsValid()

    # 4. Free Edge 존재 여부 (TopoDS_Compound → TopExp_Explorer로 wire 수 계산)
    sa = ShapeAnalysis_FreeBounds(shape)
    closed_wires_compound = sa.GetClosedWires()
    open_wires_compound   = sa.GetOpenWires()
    info["n_free_closed_wires"] = _count_wires_in_compound(closed_wires_compound)
    info["n_free_open_wires"]   = _count_wires_in_compound(open_wires_compound)
    info["no_free_edges"]       = (info["n_free_open_wires"] == 0)

    # 5. Shell 개수 및 각 Shell Closed 여부
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    shells_closed = []
    while exp.More():
        shell = exp.Current()
        shells_closed.append(shell.Closed())
        exp.Next()
    info["n_shells"] = len(shells_closed)
    info["shells_all_closed"] = all(shells_closed) if shells_closed else False

    # 종합 판정
    is_watertight = (
        info["is_solid"] and
        info["is_valid"] and
        info["no_free_edges"] and
        info["shells_all_closed"]
    )

    return is_watertight, info

def make_solid(shape):
    """Compound/Shell에서 Solid 추출 — null triangulation face 필터링 포함"""
    exp_shell = TopExp_Explorer(shape, TopAbs_SHELL)
    maker = BRepBuilderAPI_MakeSolid()
    while exp_shell.More():
        orig_shell = topods.Shell(exp_shell.Current())
        cleaned_shell = _rebuild_shell_repairing_null_faces(orig_shell)
        maker.Add(cleaned_shell)
        exp_shell.Next()
    maker.Build()
    if maker.IsDone():
        return maker.Solid()
    return shape


def _process_single_npz(npz_path, out_step_path, result_queue):
    """각 npz 파일을 별도 프로세스에서 처리 (MakerVolume hang 방지용).
    결과를 result_queue에 'watertight' / 'fail' / 'error' 문자열로 저장.
    """
    try:
        npz_data = dict(np.load(npz_path))
        uvgrid = UvGrid.load_from_npz_data(npz_data)
        print(f"  UVGrid loaded: {uvgrid.coord.shape}")
        n_valid = int((~uvgrid.empty_mask).sum())
        print(f"  Valid faces: {n_valid}")

        brep = try_combined_bspline_occ(uvgrid, verbose=True, timeout=PROCESS_TIMEOUT_SEC)
        if brep is None:
            result_queue.put("fail")
            return

        brep = heal_shape(brep, tolerance=1e-6, verbose=True, timeout=PROCESS_TIMEOUT_SEC)
        if brep is None:
            print("  ERROR: heal_shape timeout/failure!")
            result_queue.put("fail")
            return

        is_watertight, info = check_watertight(brep)
        if is_watertight:
            print("  Watertight: True")
            write_step_file(brep, out_step_path)
            print(f"  STEP written: {out_step_path}")
            result_queue.put("watertight")
        else:
            result_queue.put("fail")
    except Exception as e:
        import traceback
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()
        result_queue.put("error")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BRep post-processing from npz UV grids")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz_dir", type=str, help="Directory containing *.npz files")
    group.add_argument("--npz_file", type=str, help="Single .npz file path")
    group.add_argument("--npz_list", type=str, help="Text file with one .npz path per line")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .step files")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.npz_dir:
        npz_files = sorted(glob(os.path.join(args.npz_dir, "*.npz")))
    elif args.npz_file:
        npz_files = [args.npz_file]
    else:
        with open(args.npz_list, "r") as f:
            npz_files = [line.strip() for line in f if line.strip()]

    print(f"Total npz files found: {len(npz_files)}")

    success_count = 0
    fail_count = 0
    watertight_count = 0

    PER_FILE_TIMEOUT = PROCESS_TIMEOUT_SEC + 30

    for npz_path in npz_files:
        stem = os.path.splitext(os.path.basename(npz_path))[0]  # e.g. "00000"
        out_step_path = os.path.join(out_dir, f"{stem}.step")

        print(f"\n[{stem}] Loading npz from: {npz_path}")

        result_queue = Queue()
        p = Process(
            target=_process_single_npz,
            args=(npz_path, out_step_path, result_queue),
        )
        p.start()
        p.join(timeout=PER_FILE_TIMEOUT)

        if p.is_alive():
            print(f"  TIMEOUT [{stem}]: {PER_FILE_TIMEOUT}s 초과, 프로세스 강제 종료")
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join()
            fail_count += 1
        else:
            try:
                result = result_queue.get_nowait()
            except Exception:
                result = "error"

            if result == "watertight":
                watertight_count += 1
                success_count += 1
            else:
                fail_count += 1

        print(f"  Progress — Success: {success_count}, Failed: {fail_count}, Watertight: {watertight_count} / {len(npz_files)}")

    print(f"\n=== Done ===")
    print(f"Success: {success_count} / {len(npz_files)}")
    print(f"Failed:  {fail_count} / {len(npz_files)}")
    print(f"Watertight: {watertight_count} / {len(npz_files)}")