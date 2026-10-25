"""用“正常 3D AABB IoU（中心 xyz ± lwh/2）”分析交叉验证 dump（只用训练标签）：
选对目标的片段里，SR@0.5 / 中心误差 按 文本高度、距离、停/动 分组；以及 dx/dy/dz 各自误差与“若某一维完美”的上限。
用法：python ana_true.py dump_f0.json ..."""
import json
import sys

import numpy as np


def iou(p, g):
    lo_p, hi_p = p[:, 3:] - p[:, :3] / 2, p[:, 3:] + p[:, :3] / 2
    lo_g, hi_g = g[:, 3:] - g[:, :3] / 2, g[:, 3:] + g[:, :3] / 2
    inter = np.clip(np.minimum(hi_p, hi_g) - np.maximum(lo_p, lo_g), 0, None).prod(1)
    vp, vg = p[:, :3].prod(1), g[:, :3].prod(1)
    return inter / (vp + vg - inter)


d = {}
for path in sys.argv[1:]:
    d.update(json.load(open(path)))
ok, bad = [], []
for v, x in d.items():
    p, g = np.array(x["pred"]), np.array(x["gt"])
    sel = np.mean([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])])
    (ok if sel >= 0.8 else bad).append((v, x, p, g))
print(f"clips {len(d)}: correctly selected {len(ok)}, others {len(bad)}")
P = np.concatenate([r[2] for r in ok])
G = np.concatenate([r[3] for r in ok])
I = iou(P, G)
E = np.linalg.norm(P[:, 3:] - G[:, 3:], axis=1)
print("correct clips, frame level: true SR@0.5 %.1f  SR@0.7 %.1f  ACE %.3f  PR(<0.1m) %.1f" %
      (100 * (I > 0.5).mean(), 100 * (I > 0.7).mean(), E.mean(), 100 * (E < 0.1).mean()))
print("  |dx| |dy| |dz| mean", np.abs(P[:, 3:] - G[:, 3:]).mean(0).round(3), "median",
      np.median(np.abs(P[:, 3:] - G[:, 3:]), 0).round(3), " |dl| |dw| |dh| mean", np.abs(P[:, :3] - G[:, :3]).mean(0).round(3))
for k, name in enumerate("lwhxyz"):
    Q = P.copy()
    Q[:, k] = G[:, k]
    print(f"  if {name} exact: SR@0.5 {100 * (iou(Q, G) > 0.5).mean():5.1f}", end="")
Q = P.copy()
Q[:, 3:] = G[:, 3:]
print(f"\n  if xyz exact: SR@0.5 {100 * (iou(Q, G) > 0.5).mean():5.1f}", end="")
Q = P.copy()
Q[:, :3] = G[:, :3]
print(f"   if lwh exact: SR@0.5 {100 * (iou(Q, G) > 0.5).mean():5.1f}")


def group(name, keyf, bins):
    print(f"by {name}:")
    for lo, hi in bins:
        rows = [r for r in ok if lo <= keyf(r) < hi]
        if not rows:
            continue
        p = np.concatenate([r[2] for r in rows])
        g = np.concatenate([r[3] for r in rows])
        i = iou(p, g)
        e = np.abs(p[:, 3:] - g[:, 3:])
        print(f"  [{lo},{hi}): clips {len(rows):3d}  SR@0.5 {100 * (i > 0.5).mean():5.1f}  SR@0.7 {100 * (i > 0.7).mean():5.1f}  "
              f"|dx| {e[:, 0].mean():.2f} |dy| {e[:, 1].mean():.2f} |dz| {e[:, 2].mean():.2f}  ACE {np.linalg.norm(e, axis=1).mean():.2f}")


group("text height h", lambda r: r[2][0, 2], [(0, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 10)])
group("distance", lambda r: np.hypot(r[3][0, 3], r[3][0, 4]), [(0, 20), (20, 30), (30, 60), (60, 90), (90, 300)])
group("parked(0)/moving(1)", lambda r: 0 if "park" in (r[1]["state"] or "").lower() else 1, [(0, 1), (1, 2)])
