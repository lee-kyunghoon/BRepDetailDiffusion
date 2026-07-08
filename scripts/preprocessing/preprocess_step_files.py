import argparse
import os
import sys
import glob
import time
import warnings
from typing import List, Optional, Dict, Union
from multiprocessing import Pool
from occwl.io import load_step
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from occwl.entity_mapper import EntityMapper
import numpy as np
import h5py
from tqdm import tqdm


# occwl 내부의 pythonocc deprecation warning 숨김
warnings.filterwarnings(
    "ignore",
    message=r"Call to deprecated function BRep_Tool_Surface.*",
    category=DeprecationWarning,
    module=r"occwl\.face",
)


# ---------------------------------------------------------------------------
# STEP → UV grid 변환 핵심 함수
# ---------------------------------------------------------------------------

def parse_step_file(
    step_path: str,
    condition_path: str,
    num_u: int = 8,
    num_v: int = 8,
) -> Optional[Dict[str, np.ndarray]]:

    try:
        solids = load_step(step_path)
        condition_solids = load_step(condition_path)
    except Exception:
        return None

    # 단일 solid만 허용
    if len(solids) != 1:
        return None
    if len(condition_solids) != 1:
        return None

    solid: Solid = solids[0]
    condition_solid: Solid = condition_solids[0]

    # Closed surface/edge 분할 (BrepDiff 방식)
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    condition_solid = condition_solid.split_all_closed_faces(num_splits=0)
    condition_solid = condition_solid.split_all_closed_edges(num_splits=0)

    face_pnts, face_normals, face_masks = [], [], []

    for face in solid.faces():
        try:
            # UV grid 포인트 샘플링
            points = uvgrid(face, method="point", num_u=num_u, num_v=num_v)
            normals = uvgrid(face, method="normal", num_u=num_u, num_v=num_v)
            visibility_status = uvgrid(
                face, method="visibility_status", num_u=num_u, num_v=num_v
            )
            # Mask: 0=Inside, 1=Outside, 2=On boundary → Inside 또는 Boundary면 유효
            mask = np.logical_or(
                visibility_status == 0, visibility_status == 2
            )

            # 모든 추출이 성공한 후에만 리스트에 추가
            face_pnts.append(points)
            face_normals.append(normals)
            face_masks.append(mask)
        except Exception:
            continue

    if len(face_pnts) == 0:
        return None

    coords = np.stack(face_pnts)       # (N, num_u, num_v, 3)
    normals_arr = np.stack(face_normals)  # (N, num_u, num_v, 3)
    masks = np.stack(face_masks).squeeze(-1)  # (N, num_u, num_v)
    n_faces = len(face_pnts)

    # Build topology condition only from condition_solid.
    mapper = EntityMapper(condition_solid)
    condition_faces = list(condition_solid.faces())
    cond_face_occ_indices = [mapper.face_index(face) for face in condition_faces]
    cond_face_occ_to_local = {
        occ_idx: local_idx for local_idx, occ_idx in enumerate(cond_face_occ_indices)
    }
    n_condition_faces = len(cond_face_occ_indices)

    face_adj = np.zeros((n_condition_faces, n_condition_faces), dtype=np.uint8)
    for edge in condition_solid.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(condition_solid.faces_from_edge(edge))
        if len(connected_faces) != 2:
            continue

        if edge.seam(connected_faces[0]) or edge.seam(connected_faces[1]):
            continue

        left_face, right_face = edge.find_left_and_right_faces(connected_faces)
        if (left_face is None) or (right_face is None):
            continue

        left_occ_idx = mapper.face_index(left_face)
        right_occ_idx = mapper.face_index(right_face)

        if (left_occ_idx not in cond_face_occ_to_local) or (
            right_occ_idx not in cond_face_occ_to_local
        ):
            continue

        i = cond_face_occ_to_local[left_occ_idx]
        j = cond_face_occ_to_local[right_occ_idx]
        if i == j:
            continue
        face_adj[i, j] = 1
        face_adj[j, i] = 1

    types = None

    coords = normalize_to_unit_cube(coords)

    return {
        "coords": coords.astype(np.float32),
        "normals": normals_arr.astype(np.float32),
        "masks": masks,
        "types": types,
        "face_adj": face_adj,
        "n_faces": n_faces,
        "n_condition_faces": n_condition_faces,
    }


def normalize_to_unit_cube(
    coords: np.ndarray,
):
    """좌표를 [-1, 1] 단위 정육면체로 정규화."""
    mins = np.min(coords.reshape(-1, 3), axis=0)
    maxs = np.max(coords.reshape(-1, 3), axis=0)
    center = (mins + maxs) / 2
    scale = np.max(maxs - mins) / 2
    if scale < 1e-10:
        scale = 1.0

    coords = (coords - center) / scale

    return coords


def process_single_step(args):
    step_path, condition_path, num_u, num_v, uid = args
    try:
        data = parse_step_file(step_path, condition_path, num_u=num_u, num_v=num_v)
        if data is None:
            return uid, None, "Failed to parse (no solid or empty faces)"
        return uid, data, None
    except Exception as e:
        return uid, None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 메인 전처리 파이프라인
# ---------------------------------------------------------------------------

def preprocess(
    step_dir: Union[str, List[str]],
    condition_dir: str,
    out_dir: str,
    dataset_name: str,
    max_n_prims: int = 50,
    n_grid: int = 8,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 16,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test 비율의 합이 1.0이어야 합니다."

    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1) STEP 파일 수집
    # ------------------------------------------------------------------
    if isinstance(step_dir, str):
        step_dirs = [step_dir]
    else:
        step_dirs = step_dir

    step_dirs = [os.path.abspath(d) for d in step_dirs]
    for d in step_dirs:
        if not os.path.isdir(d):
            print(f"[ERROR] STEP 디렉토리가 존재하지 않습니다: '{d}'")
            sys.exit(1)

    step_patterns = ["*.step", "*.stp", "*.STEP", "*.STP"]
    step_files = []
    for root_dir in step_dirs:
        for pattern in step_patterns:
            step_files.extend(
                glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
            )
    step_files = sorted(list(set(step_files)))

    if len(step_files) == 0:
        print(f"[ERROR] '{step_dirs}'에서 STEP 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"총 {len(step_files)}개의 STEP 파일을 찾았습니다.")

    # uid 충돌 방지: 루트 기준 상대경로를 uid로 사용
    def _make_uid(step_path: str) -> str:
        step_path_abs = os.path.abspath(step_path)
        best_root = None
        best_len = -1
        for root in step_dirs:
            try:
                common = os.path.commonpath([step_path_abs, root])
            except ValueError:
                continue
            if common == root and len(root) > best_len:
                best_root = root
                best_len = len(root)

        if best_root is None:
            rel = os.path.basename(step_path_abs)
        else:
            rel = os.path.relpath(step_path_abs, best_root)

        rel_no_ext = os.path.splitext(rel)[0]
        uid = rel_no_ext.replace("\\", "/").replace("/", "__")
        return uid

    # 최종 uid 중복 체크
    uid_counts: Dict[str, int] = {}
    worker_args = []
    for p in step_files:
        uid = _make_uid(p)
        uid_counts[uid] = uid_counts.get(uid, 0) + 1

        if "bracket" in uid:
            condition_path = os.path.join(condition_dir, "bracket/bracket_concept.step")
        elif "bush" in uid:
            condition_path = os.path.join(condition_dir, "bush/bush_concept.step")
        elif "shaft" in uid:
            condition_path = os.path.join(condition_dir, "shaft/shaft_concept.step")
        elif "washer" in uid:
            condition_path = os.path.join(condition_dir, "washer/washer_concept.step")
        elif "part_part1" in uid:
            condition_path = os.path.join(condition_dir, "part1_1_concept.step")
        elif "part_part2" in uid:
            condition_path = os.path.join(condition_dir, "part2_1_concept.step")
        elif "part_part3" in uid:
            condition_path = os.path.join(condition_dir, "part3_1_concept.step")
        elif "part1" in uid:
            condition_path = os.path.join(condition_dir, "part1_concept.step")
        elif "part2" in uid:
            condition_path = os.path.join(condition_dir, "part2_concept.step")
        elif "part3" in uid:
            condition_path = os.path.join(condition_dir, "part3_concept.step")
        elif "part4" in uid:
            condition_path = os.path.join(condition_dir, "part4_concept.step")
        elif "part5" in uid:
            condition_path = os.path.join(condition_dir, "part5_concept.step")
        elif "part6" in uid:
            condition_path = os.path.join(condition_dir, "part6_concept.step")
        elif "part7" in uid:
            condition_path = os.path.join(condition_dir, "part7_concept.step")
        elif "part8" in uid:
            condition_path = os.path.join(condition_dir, "part8_concept.step")
        elif "part9" in uid:
            condition_path = os.path.join(condition_dir, "part9_concept.step")
        else:
            condition_path = os.path.join(condition_dir, "supporter/supporter_concept.step")

        if not os.path.exists(condition_path):
            print(f"[ERROR] condition STEP 파일이 존재하지 않습니다: '{condition_path}'")
            sys.exit(1)

        worker_args.append((p, condition_path,  n_grid, n_grid, uid))

    dup_uids = [u for u, c in uid_counts.items() if c > 1]
    if len(dup_uids) > 0:
        print(f"[ERROR] uid 충돌이 발생했습니다. (예: {dup_uids[:3]})")
        print("STEP 파일 경로가 uid로 유일해야 합니다. 입력 디렉토리 구성을 확인해주세요.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2) 병렬 처리로 UV grid 추출
    # ------------------------------------------------------------------
    h5_path = os.path.join(out_dir, f"{dataset_name}_grid{n_grid}.h5")
    log_lines = []
    success_uids = []
    fail_uids = []
    n_faces_list = []

    print(f"UV grid 추출 중 (workers={num_workers})...")
    start_time = time.time()

    with h5py.File(h5_path, "w") as h5f:
        data_grp = h5f.create_group("data")

        if num_workers <= 1:
            results = [process_single_step(a) for a in tqdm(worker_args)]
        else:
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_single_step, worker_args),
                        total=len(worker_args),
                    )
                )

        for uid, data, err in results:
            if data is None:
                fail_uids.append(uid)
                log_lines.append(f"FAIL  {uid}: {err}")
                continue

            n_faces = data["n_faces"]
            n_condition_faces = data["n_condition_faces"]

            # face 수 필터링
            if n_faces > max_n_prims:
                fail_uids.append(uid)
                log_lines.append(
                    f"SKIP  {uid}: n_faces={n_faces} > max_n_prims={max_n_prims}"
                )
                continue

            if n_condition_faces > max_n_prims:
                fail_uids.append(uid)
                log_lines.append(
                    f"SKIP  {uid}: n_condition_faces={n_condition_faces} > max_n_prims={max_n_prims}"
                )
                continue

            # H5에 저장
            grp = data_grp.create_group(uid)
            grp.create_dataset("coords", data=data["coords"], compression="gzip")
            grp.create_dataset("normals", data=data["normals"], compression="gzip")
            grp.create_dataset("masks", data=data["masks"], compression="gzip")
            grp.create_dataset("face_adj", data=data["face_adj"], compression="gzip")

            success_uids.append(uid)
            n_faces_list.append(n_faces)
            log_lines.append(f"OK    {uid}: n_faces={n_faces}")

    elapsed = time.time() - start_time
    print(f"UV grid 추출 완료: {len(success_uids)}/{len(step_files)} 성공 ({elapsed:.1f}s)")
    print(f"  실패/스킵: {len(fail_uids)}개")

    if len(success_uids) == 0:
        print("[ERROR] 성공적으로 처리된 파일이 없습니다. 전처리를 중단합니다.")
        sys.exit(1)

    n_total = len(success_uids)
    indices = np.random.permutation(n_total)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # 나머지는 test
    n_test = n_total - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        "train": [success_uids[i] for i in train_indices],
        "val": [success_uids[i] for i in val_indices],
        "test": [success_uids[i] for i in test_indices],
    }

    print(f"데이터 분할: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # 분할 파일 저장
    for split_name, uid_list in splits.items():
        txt_path = os.path.join(out_dir, f"{dataset_name}_{max_n_prims}_{split_name}.txt")
        with open(txt_path, "w") as f:
            for uid in sorted(uid_list):
                f.write(f"{uid}\n")
        print(f"  → {txt_path} ({len(uid_list)}개)")

    # 호환성을 위한 빈 pkl_absence 파일
    absence_path = os.path.join(out_dir, f"{dataset_name}_{max_n_prims}_pkl_absence.txt")
    with open(absence_path, "w") as f:
        pass  # 빈 파일
    print(f"  → {absence_path} (빈 파일)")

    train_n_faces = np.array([n_faces_list[i] for i in train_indices])
    n_faces_path = os.path.join(out_dir, f"{dataset_name}_{max_n_prims}_train_n_faces.npz")
    np.savez(n_faces_path, n_faces=train_n_faces)
    print(f"  → {n_faces_path} (train face 수 분포)")

    log_path = os.path.join(out_dir, "preprocess_log.txt")
    with open(log_path, "w") as f:
        f.write(f"# Preprocessing log\n")
        f.write(f"# step_dir: {step_dirs}\n")
        f.write(f"# n_grid: {n_grid}\n")
        f.write(f"# max_n_prims: {max_n_prims}\n")
        f.write(f"# total: {len(step_files)}, success: {len(success_uids)}, fail: {len(fail_uids)}\n")
        f.write(f"# train: {n_train}, val: {n_val}, test: {n_test}\n")
        f.write(f"# seed: {seed}\n\n")
        for line in log_lines:
            f.write(line + "\n")
    print(f"  → {log_path}")

    print("\n" + "=" * 70)
    print("전처리 완료!")
    print("=" * 70)
    print(f"\n학습 시 config 파일에서 다음 값들을 설정하세요:\n")
    print(f"  dataset: {dataset_name}")
    print(f"  h5_path: '{h5_path}'")
    print(f"  n_grid: {n_grid}")
    print(f"  max_n_prims: {max_n_prims}")
    if os.path.exists(n_faces_path):
        print(f"  n_faces_path: '{n_faces_path}'")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="STEP 파일을 BrepDiff 학습용 HDF5 데이터로 전처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step-dir", nargs="+", type=str, default=["./LinearMotionGuide"],
        help="STEP 파일들이 있는 디렉토리 경로(여러 개 지정 가능). 각 경로에서 재귀 탐색",
    )
    parser.add_argument(
        "--condition-dir", type=str, default="./conditions",
        help="조건 파일들이 있는 디렉토리 경로",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./data/processed",
        help="전처리 결과 출력 디렉토리",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="custom",
        help="데이터셋 이름 (출력 파일명 prefix, 기본: custom)",
    )
    parser.add_argument(
        "--max-n-prims", type=int, default=60,
        help="최대 허용 face 수. 초과 시 필터링 (기본: 60)",
    )
    parser.add_argument(
        "--n-grid", type=int, default=10,
        help="UV grid 해상도 (기본: 16, BrepDiff 기본값)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="학습 데이터 비율 (기본: 0.8)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="검증 데이터 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="테스트 데이터 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=20,
        help="병렬 처리 워커 수 (기본: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    args = parser.parse_args()

    preprocess(
        step_dir=args.step_dir,
        condition_dir=args.condition_dir,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        max_n_prims=args.max_n_prims,
        n_grid=args.n_grid,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
