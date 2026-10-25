"""已知文本尺寸 (l,w,h)、航向（文本 azimuth / 朝向词）和文本距离时，按“3D 框投影外接矩形 ≈ 检测框”求目标中心。

用途：近处目标的框中心并不是 3D 中心的投影（透视），被图像边界截断的框更是如此；
只用未截断的边做约束，就能把截断目标的中心推回到图像外侧。
参数化：中心 (x, y) = r (cos θ, sin θ)（r 为文本距离，水平距离），未知量为 θ 与 z；
代价 = Σ_未截断边 ((投影矩形边 - 检测框边 - bias·尺寸) / (sig·尺寸 + 1px))² + ((z - z先验)/zsig)²。
"""
import numpy as np

SIGNS = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], dtype=np.float64)


def corners(centers, lwh, yaw):
    """centers (N,3) -> (N,8,3)。l 沿航向 yaw（virtuallidar 下，0 = +x）。"""
    off = SIGNS * (np.asarray(lwh, dtype=np.float64) / 2)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.stack([off[:, 0] * c - off[:, 1] * s, off[:, 0] * s + off[:, 1] * c, off[:, 2]], 1)
    return np.asarray(centers, dtype=np.float64)[:, None, :] + rot[None]


def rects(centers, lwh, yaw, calib):
    """3D 框投影外接矩形 (N,4) = (umin, vmin, umax, vmax)，以及是否全部角点在相机前方。"""
    P = corners(centers, lwh, yaw).reshape(-1, 3)
    cam = P @ calib["R"].T + calib["t"]
    z = cam[:, 2]
    K = calib["K"]
    zs = np.where(z > 0.1, z, 0.1)
    u = (K[0, 0] * cam[:, 0] / zs + K[0, 2]).reshape(-1, 8)
    v = (K[1, 1] * cam[:, 1] / zs + K[1, 2]).reshape(-1, 8)
    valid = (z.reshape(-1, 8) > 0.5).all(1)
    return np.stack([u.min(1), v.min(1), u.max(1), v.max(1)], 1), valid


def trunc_flags(box, margin, w_img=1920, h_img=1080):
    return np.array([box[0] < margin, box[1] < margin, box[2] > w_img - margin, box[3] > h_img - margin])


def fit_center(box, trunc, lwh, yaw, r, calib, plane, hc, zsig, bias, sig, theta0, half_lat=4.0, z_half=1.5,
               side_w=None):
    """返回 (center(3), cost, n_used)。plane=(a,b,c)：地面 z = a x + b y + c；z 先验 = 地面 + hc。
    bias/sig：长度 4，单位为框宽（u 边）或框高（v 边）。side_w：各边权重（默认 1）。"""
    bw, bh = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
    S = np.array([bw, bh, bw, bh])
    use = ~np.asarray(trunc, dtype=bool)
    if not use.any():
        return None, np.inf, 0
    wts = np.ones(4) if side_w is None else np.asarray(side_w, dtype=np.float64)
    target = np.asarray(box[:4], dtype=np.float64) + np.asarray(bias) * S
    sd = np.asarray(sig) * S + 1.0
    a, b, c = plane
    th_c, th_h = theta0, half_lat / max(r, 1.0)
    z_c = None
    z_h = z_half
    best = None
    for level in range(3):
        ths = np.linspace(th_c - th_h, th_c + th_h, 41)
        xs, ys = r * np.cos(ths), r * np.sin(ths)
        zp = a * xs + b * ys + c + hc  # 每个 θ 的 z 先验
        if z_c is None:
            zoff = np.linspace(-z_h, z_h, 31)
            Z = zp[:, None] + zoff[None, :]
        else:
            Z = np.broadcast_to(np.linspace(z_c - z_h, z_c + z_h, 31)[None, :], (len(ths), 31))
        C = np.stack([np.repeat(xs, Z.shape[1]), np.repeat(ys, Z.shape[1]), Z.ravel()], 1)
        R, valid = rects(C, lwh, yaw, calib)
        res = (R - target) / sd
        cost = (wts[use] * res[:, use] ** 2).sum(1) + ((C[:, 2] - np.repeat(zp, Z.shape[1])) / zsig) ** 2
        cost = np.where(valid, cost, np.inf)
        k = int(np.argmin(cost))
        if not np.isfinite(cost[k]):
            return None, np.inf, int(use.sum())
        best = (C[k].copy(), float(cost[k]))
        th_c = float(np.arctan2(C[k, 1], C[k, 0]))
        z_c = float(C[k, 2])
        th_h = th_h / 8.0
        z_h = z_h / 8.0
    return best[0], best[1], int(use.sum())
