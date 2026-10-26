"""几何精度诊断（只用 TRAIN 标签，选择用 GT 对应的检测框 = oracle，单独看中心估计本身的误差）。
A. 各相机 GT 中心离全局地平面的高度 hc，与文本高度 h 的关系。
B. 3D 框投影外接矩形 vs YOLO 检测框：各边偏置（框尺寸归一化），检验航向约定（yaw vs yaw+90°）。
C. 中心估计对比：ray（v11c 做法）vs boxfit（boxfit.py），按截断/距离/相机分组，逐帧不平滑。
用法：python ana_geo.py [every]"""
import collections
import json
import sys

import numpy as np

from boxfit import fit_center, rects, trunc_flags
from geo_common import WORK, load_calib, load_clips, parse_text, pixel_rays, project

EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DETS = json.load(open(WORK + "/cache/dets_yolo11x_1920.json"))
W_IMG, H_IMG = 1920, 1080
clips = load_clips("train")
TEST_SEQS = {"0000", "0001", "0002", "0008", "0029", "0033", "0057", "0059"}


def contains(b, uv):
    """像素点 uv 是否落在框 b 内（含边界）。"""
    return b[0] <= uv[0] <= b[2] and b[1] <= uv[1] <= b[3]


def iou2d(a, b):
    """两个 2D 框 (x1, y1, x2, y2, ...) 的 IoU。"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def iou3d(lwh_p, c_p, lwh_g, c_g):
    """单个预测框与 GT 框（尺寸 + 中心）的轴对齐 3D IoU。"""
    lo = np.maximum(c_p - lwh_p / 2, c_g - lwh_g / 2)
    hi = np.minimum(c_p + lwh_p / 2, c_g + lwh_g / 2)
    inter = np.clip(hi - lo, 0, None).prod()
    return inter / (lwh_p.prod() + lwh_g.prod() - inter)


# 全局地平面（与 pipeline3 --ground global 一致）
rows = np.array([[f["xyz"][0], f["xyz"][1], f["xyz"][2] - f["lwh"][2] / 2] for c in clips for f in c["frames"]
                 if f["xyz"] is not None])
plane = np.linalg.lstsq(np.c_[rows[:, 0], rows[:, 1], np.ones(len(rows))], rows[:, 2], rcond=None)[0]
print("global plane", plane.round(4))


def range_curve(dist):
    """文本距离 → 逐帧距离（关键帧 0 / 14 / 29 之间分段线性插值，同 pipeline3.range_curve 的 lin 方式）。"""
    if len(dist) >= 3:
        return np.interp(np.arange(30), [0, 14, 29], dist[:3])
    if len(dist) == 2:
        return np.interp(np.arange(30), [0, 29], dist[:2])
    if len(dist) == 1:
        return np.full(30, dist[0])
    return None


def match(clip, fi, lwh_t, yaw):
    """center 匹配；失败则用 GT 3D 框（文本尺寸+航向）投影矩形（裁到图内）的最大 IoU 框。返回 (box, how)。"""
    f = clip["frames"][fi]
    calib = load_calib(f["img"])
    boxes = DETS.get(f"train/{f['img']}", [])
    uv, _ = project(f["xyz"][None], calib)
    hits = [b for b in boxes if contains(b, uv[0])]
    if hits:
        return min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), "center"
    R, valid = rects(f["xyz"][None], lwh_t, yaw, calib)
    if not valid[0]:
        return None, None
    r = [max(0, R[0, 0]), max(0, R[0, 1]), min(W_IMG, R[0, 2]), min(H_IMG, R[0, 3])]
    if r[2] <= r[0] or r[3] <= r[1]:
        return None, None
    best = max(boxes, key=lambda b: iou2d(b, r), default=None)
    if best is None or iou2d(best, r) < 0.3:
        return None, None
    return best, "rect"


# ---------------------------------------------------------------- 收集样本
samples = []
for c in clips:
    t = parse_text(c["desc"])
    if t["lwh"] is None:
        continue
    rc = range_curve(t["distances"])
    if rc is None:
        continue
    yaw = np.radians(t["azimuth"] if t["azimuth"] is not None else 0.0)
    for fi in range(30):
        f = c["frames"][fi]
        if f["xyz"] is None:
            continue
        b, how = match(c, fi, t["lwh"], yaw)
        if b is None:
            continue
        tr = trunc_flags(b, 4.0)
        if not tr.any() and fi % EVERY:
            continue
        samples.append(dict(clip=c, fi=fi, box=b, how=how, trunc=tr, lwh=t["lwh"], yaw=yaw, r=rc[fi],
                            seq=c["seq"], park="park" in (c["state"] or "").lower()))
print("samples", len(samples), "truncated", sum(s["trunc"].any() for s in samples),
      "rect-matched", sum(s["how"] == "rect" for s in samples))

# ---------------------------------------------------------------- A. GT 中心高度
print("\nA. GT centre height above global plane, per camera (median hc, median h, hc-h/2)")
by = collections.defaultdict(list)
for s in samples:
    g = s["clip"]["frames"][s["fi"]]
    zp = plane[0] * g["xyz"][0] + plane[1] * g["xyz"][1] + plane[2]
    by[s["seq"]].append((g["xyz"][2] - zp, g["lwh"][2], s["clip"]["type"]))
for seq in sorted(by):
    v = np.array([x[:2] for x in by[seq]])
    types = collections.Counter(x[2] for x in by[seq])
    print(f"  {seq}{'*' if seq in TEST_SEQS else ' '} n={len(v):4d} hc {np.median(v[:, 0]):6.3f}  h {np.median(v[:, 1]):5.2f}"
          f"  hc-h/2 {np.median(v[:, 0] - v[:, 1] / 2):6.3f}  types {dict(types)}")
allv = np.array([(x[0], x[1]) for v in by.values() for x in v])
A = np.c_[allv[:, 1], np.ones(len(allv))]
coef = np.linalg.lstsq(A, allv[:, 0], rcond=None)[0]
print("  all: hc ~ %.3f*h + %.3f" % tuple(coef))

# ---------------------------------------------------------------- B. 投影矩形 vs 检测框
print("\nB. side residual (proj rect - det box)/size on non-truncated GT-centre samples")
res = {"yaw": [], "yaw90": []}
for s in samples:
    if s["trunc"].any():
        continue
    g = s["clip"]["frames"][s["fi"]]
    calib = load_calib(g["img"])
    b = np.array(s["box"][:4])
    S = np.array([b[2] - b[0], b[3] - b[1], b[2] - b[0], b[3] - b[1]])
    for key, yaw in (("yaw", s["yaw"]), ("yaw90", s["yaw"] + np.pi / 2)):
        R, ok = rects(g["xyz"][None], s["lwh"], yaw, calib)
        if ok[0]:
            res[key].append(np.r_[(R[0] - b) / S, s["r"]])
for key in res:
    v = np.array(res[key])
    print(f"  {key:6s}: median {np.median(v[:, :4], 0).round(3)}  MAD {np.median(np.abs(v[:, :4] - np.median(v[:, :4], 0)), 0).round(3)}")
v = np.array(res["yaw"])
BIAS = np.median(v[:, :4], 0)
SIG = 1.4826 * np.median(np.abs(v[:, :4] - BIAS), 0)
for lo, hi in [(0, 25), (25, 45), (45, 80), (80, 300)]:
    m = (v[:, 4] >= lo) & (v[:, 4] < hi)
    if m.any():
        print(f"    r [{lo},{hi}) n={m.sum():4d} median {np.median(v[m, :4], 0).round(3)}")

# ---------------------------------------------------------------- C. 中心估计对比
offs = []
for s in samples:
    if s["trunc"].any():
        continue
    g = s["clip"]["frames"][s["fi"]]
    uv, _ = project(g["xyz"][None], load_calib(g["img"]))
    b = s["box"]
    offs.append([(uv[0, 0] - (b[0] + b[2]) / 2) / (b[2] - b[0]), (uv[0, 1] - (b[1] + b[3]) / 2) / (b[3] - b[1])])
OFFSET = np.median(np.array(offs), 0)
print("\nglobal box offset", OFFSET.round(4), " bias", BIAS.round(3), " sig", SIG.round(3))


def on_range(o, d, r):
    """射线与竖直圆柱面 x² + y² = r² 的交点，取较大的根；无交点或在相机后方时返回 None（同 pipeline3.on_range）。"""
    a = d[0] ** 2 + d[1] ** 2
    bb = 2 * (o[0] * d[0] + o[1] * d[1])
    cc = o[0] ** 2 + o[1] ** 2 - r ** 2
    disc = bb * bb - 4 * a * cc
    if disc < 0:
        return None
    sol = (-bb + np.sqrt(disc)) / (2 * a)
    return o + sol * d if sol > 0 else None


out = []
for s in samples:
    g = s["clip"]["frames"][s["fi"]]
    calib = load_calib(g["img"])
    b = s["box"]
    w, h = b[2] - b[0], b[3] - b[1]
    o, d = pixel_rays([[(b[0] + b[2]) / 2 + OFFSET[0] * w, (b[1] + b[3]) / 2 + OFFSET[1] * h]], calib)
    p = on_range(o, d[0], s["r"])
    if p is None:
        continue
    lwh = s["lwh"]
    zpl = plane[0] * p[0] + plane[1] * p[1] + plane[2] + lwh[2] / 2
    e0 = np.array([p[0], p[1], 0.75 * p[2] + 0.25 * zpl])
    th0 = np.arctan2(p[1], p[0])
    est = {"ray": e0}
    for name, kw in [("bf", dict(bias=BIAS, sig=SIG, hc=lwh[2] / 2, zsig=0.3)),
                     ("bf_nobias", dict(bias=np.zeros(4), sig=SIG, hc=lwh[2] / 2, zsig=0.3)),
                     ("bf_zs1", dict(bias=BIAS, sig=SIG, hc=lwh[2] / 2, zsig=1.0)),
                     ("bf_zs.1", dict(bias=BIAS, sig=SIG, hc=lwh[2] / 2, zsig=0.1))]:
        cen, cost, n = fit_center(b, s["trunc"], lwh, s["yaw"], s["r"], calib, plane, theta0=th0, **kw)
        est[name] = cen if cen is not None else e0
    # 混合：未截断用 ray 的 z，boxfit 的 xy
    est["bf_xy+ray_z"] = np.r_[est["bf"][:2], e0[2]]
    row = {"seq": s["seq"], "r": s["r"], "trunc": bool(s["trunc"].any()), "park": s["park"], "how": s["how"],
           "vid": s["clip"]["vid"]}
    for k, e in est.items():
        row[k] = (e - g["xyz"], iou3d(lwh, e, g["lwh"], g["xyz"]))
    out.append(row)

names = [k for k in out[0] if k not in ("seq", "r", "trunc", "park", "how", "vid")]


def report(title, rs):
    """打印一组样本上各种中心估计方式的 SR@0.5、平均 IoU 与平均中心误差。"""
    if not rs:
        return
    print(f"  {title:28s} n={len(rs):5d}", end="")
    for k in names:
        e = np.array([r[k][0] for r in rs])
        i = np.array([r[k][1] for r in rs])
        print(f" | {k}: SR {100 * (i > 0.5).mean():5.1f} IoU {i.mean():.3f} |e| {np.linalg.norm(e, axis=1).mean():.3f}", end="")
    print()


print("\nC. oracle-box centre estimates (per frame, no smoothing)")
report("all", out)
report("not truncated", [r for r in out if not r["trunc"]])
report("truncated", [r for r in out if r["trunc"]])
report("truncated, rect-matched", [r for r in out if r["trunc"] and r["how"] == "rect"])
for lo, hi in [(0, 20), (20, 25), (25, 35), (35, 60), (60, 90), (90, 300)]:
    report(f"r [{lo},{hi}) not trunc", [r for r in out if lo <= r["r"] < hi and not r["trunc"]])
for seq in sorted(TEST_SEQS):
    report(f"seq {seq} not trunc", [r for r in out if r["seq"] == seq and not r["trunc"]])
print("\nsigned mean error (dx dy dz) by group:")
for title, rs in [("r<25 not trunc", [r for r in out if r["r"] < 25 and not r["trunc"]]),
                  ("r>=25 not trunc", [r for r in out if r["r"] >= 25 and not r["trunc"]]),
                  ("truncated", [r for r in out if r["trunc"]])]:
    print(f"  {title}: " + "  ".join(f"{k} {np.mean([r[k][0] for r in rs], 0).round(3)}" for k in ("ray", "bf", "bf_nobias")))
print("\ntruncated clips (per clip mean):")
tv = collections.defaultdict(list)
for r in out:
    if r["trunc"]:
        tv[r["vid"]].append(r)
for v, rs in sorted(tv.items()):
    print(f"  {v} seq {rs[0]['seq']} r {rs[0]['r']:5.1f} n {len(rs):2d} " + "  ".join(
        f"{k}: IoU {np.mean([x[k][1] for x in rs]):.2f} e {np.mean([x[k][0] for x in rs], 0).round(2)}" for k in ("ray", "bf")))
