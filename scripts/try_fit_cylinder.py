import numpy as np

def compute_circular_angle_span(angles):
    """
    [-π, π] 범위의 각도 배열에서 wrap-around를 고려한
    실제 angular span을 계산한다.
    """
    if len(angles) < 2:
        return 0.0
    angles_sorted = np.sort(angles)
    gaps = np.diff(angles_sorted)
    wrap_gap = (2.0 * np.pi) - angles_sorted[-1] + angles_sorted[0]
    gaps = np.append(gaps, wrap_gap)
    max_gap = np.max(gaps)
    return float(2.0 * np.pi - max_gap)


def fit_circle_2d(coords_2d):
    """
    2D 점들에 대해 최소자승 원 피팅.
    Returns: (cx, cy, radius, mean_residual) 또는 None
    """
    A_mat = np.column_stack([coords_2d, np.ones(len(coords_2d))])
    b_vec = (coords_2d ** 2).sum(axis=1)
    result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    params = result[0]

    cx = params[0] / 2.0
    cy = params[1] / 2.0
    r_sq = params[2] + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return None
    radius = np.sqrt(r_sq)
    if radius < 1e-6:
        return None

    dists = np.sqrt((coords_2d[:, 0] - cx) ** 2 + (coords_2d[:, 1] - cy) ** 2)
    residuals = np.abs(dists - radius)
    mean_resid = float(np.mean(residuals))

    return cx, cy, radius, mean_resid


# ================================================================
# 축 후보별 피팅
# ================================================================

def _try_fit_with_axis(pts_c, centroid, axis_dir, e1, e2):
    # 축 방향 성분 제거 → 횡단면에 투영
    proj = pts_c - np.outer(pts_c @ axis_dir, axis_dir)
    coords_2d = np.column_stack([proj @ e1, proj @ e2])

    # 원 피팅
    fit = fit_circle_2d(coords_2d)
    if fit is None:
        return None
    cx, cy, radius, mean_resid = fit

    # 각도 범위 (circular wrap-around 처리)
    centered_2d = coords_2d - np.array([cx, cy])
    angles = np.arctan2(centered_2d[:, 1], centered_2d[:, 0])
    angle_span = compute_circular_angle_span(angles)

    # face 대각선 길이
    pts = pts_c + centroid
    face_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))

    if angle_span < 0.4:
        return None

    if angle_span > 0.5:
        max_r_diag_ratio = 2.0
    else:
        max_r_diag_ratio = 5.0
    if radius > max_r_diag_ratio * face_diag:
        return None

    if mean_resid > 0.45 * radius:
        return None

    sagitta = radius * (1.0 - np.cos(angle_span / 2.0))
    if sagitta < 0.01 * face_diag:
        return None
    
    dists_from_center = np.linalg.norm(
        coords_2d - coords_2d.mean(axis=0), axis=1
    )
    plane_resid = float(np.std(dists_from_center))
    if plane_resid > 1e-12 and mean_resid > 0.8 * plane_resid:
        return None

    center_3d = centroid + cx * e1 + cy * e2
    return center_3d, axis_dir, radius, mean_resid, angle_span

def try_fit_cylinder_params_from_points(pts_flat):

    pts = np.asarray(pts_flat, dtype=np.float64)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid

    # ── PCA ──
    cov = np.cov(pts_c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eigenvalues)

    ev_min = eigenvalues[sorted_idx[0]]
    ev_mid = eigenvalues[sorted_idx[1]]
    ev_max = eigenvalues[sorted_idx[2]]

    if ev_max < 1e-3:
        return None

    if ev_mid / ev_max < 0.001:
        return None

    flatness = ev_min / ev_max
    planarity = ev_mid / ev_max
    if flatness < 0.02 and planarity > 0.3:
        return None
    
    candidates = []

    for axis_idx in range(3):
        other = [k for k in range(3) if k != axis_idx]
        axis_dir = eigenvectors[:, sorted_idx[axis_idx]]
        e1 = eigenvectors[:, sorted_idx[other[0]]]
        e2 = eigenvectors[:, sorted_idx[other[1]]]

        result = _try_fit_with_axis(pts_c, centroid, axis_dir, e1, e2)
        if result is not None:
            candidates.append(result)

    if not candidates:
        return None

    # residual이 가장 작은 후보 선택
    best = min(candidates, key=lambda c: c[3])
    center_3d, axis_dir, radius, _, _ = best

    if radius > 1.2:
        return None
    
    return center_3d, axis_dir, radius, pts
