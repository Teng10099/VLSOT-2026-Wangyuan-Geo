"""运动目标轨迹拟合（后处理）：逐帧观测给出“相机→目标”的水平方位（来自 boxfit/ray 中心），
文本在第 0/14/29 帧给出距离；拟合 p(t) = a + b·τ + c·τ²（水平二维），使
  Σ_t (观测方位线到 p(t) 的横向距离 / σ_lat)² + Σ_k ((|p(t_k)| - r_k) / σ_r)²
最小。比“距离按时间插值”更符合匀速/匀加速运动（横穿目标的距离随时间不是二次曲线）。"""
import numpy as np


def fit_traj(obs_xy, cam_xy, keys, ranges, sig_lat=None, sig_r=0.03, deg=2, iters=15, sig_acc=None):
    """obs_xy (30,2) 观测中心；cam_xy (30,2) 相机光心（virtuallidar 水平坐标）；keys 关键帧下标；ranges 文本距离。
    返回 (30,2) 拟合位置。"""
    n = len(obs_xy)
    tau = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    V = np.stack([tau ** k for k in range(deg + 1)], 1)  # (n, deg+1)
    d = obs_xy - cam_xy
    dist = np.linalg.norm(d, axis=1)
    u = d / np.maximum(dist[:, None], 1e-6)
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)  # 方位线的法向
    if sig_lat is None:
        sig_lat = 0.03 + 0.0015 * dist
    sig_lat = np.broadcast_to(sig_lat, (n,))
    # 初值：对观测点做多项式最小二乘
    coef = np.linalg.lstsq(V, obs_xy, rcond=None)[0]  # (deg+1, 2)
    for _ in range(iters):
        P = V @ coef
        rows, rhs = [], []
        # 横向：nrm_t · (P_t - cam_t) = 0（线性）
        for t in range(n):
            J = np.zeros(2 * (deg + 1))
            J[0::2] = nrm[t, 0] * V[t]
            J[1::2] = nrm[t, 1] * V[t]
            rows.append(J / sig_lat[t])
            rhs.append(-(nrm[t] @ (P[t] - cam_xy[t])) / sig_lat[t] + J @ coef.ravel() / sig_lat[t])
        # 距离：|P_k| = r_k（线性化）
        for k, r in zip(keys, ranges):
            pk = P[k]
            nk = np.linalg.norm(pk)
            g = pk / max(nk, 1e-6)
            J = np.zeros(2 * (deg + 1))
            J[0::2] = g[0] * V[k]
            J[1::2] = g[1] * V[k]
            rows.append(J / sig_r)
            rhs.append((r - nk) / sig_r + J @ coef.ravel() / sig_r)
        if sig_acc is not None and deg >= 2:
            for j in range(2, deg + 1):
                for ax in range(2):
                    J = np.zeros(2 * (deg + 1))
                    J[2 * j + ax] = 1.0 / sig_acc
                    rows.append(J)
                    rhs.append(0.0)
        A, bvec = np.array(rows), np.array(rhs)
        new = np.linalg.lstsq(A, bvec, rcond=None)[0].reshape(deg + 1, 2)
        if np.abs(new - coef).max() < 1e-6:
            coef = new
            break
        coef = new
    return V @ coef
