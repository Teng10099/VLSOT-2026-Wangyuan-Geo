"""中心精度分析（交叉验证 dump，正常 3D IoU 口径，只用 TRAIN 标签）：
停车目标：预测（常数）相对 GT 片段中位数的偏差 = 每段的系统偏差；按轴、按距离段统计，并与 GT 自身抖动比较。
运动目标：逐帧误差拆成径向 / 横向 / 高度。
用法：python ana_prec.py "runs/ablation/a9_v12f_f?.json" """
import glob
import json
import sys

import numpy as np

d = {}
for p in sorted(glob.glob(sys.argv[1])):
    d.update(json.load(open(p)))


def iou(p, g):
    lo = np.maximum(p[:, 3:] - p[:, :3] / 2, g[:, 3:] - g[:, :3] / 2)
    hi = np.minimum(p[:, 3:] + p[:, :3] / 2, g[:, 3:] + g[:, :3] / 2)
    inter = np.clip(hi - lo, 0, None).prod(1)
    return inter / (p[:, :3].prod(1) + g[:, :3].prod(1) - inter)


park_bias, park_rows, mov = [], [], []
for v, x in d.items():
    sel = np.mean([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])])
    if sel < 0.8:
        continue
    p, g = np.array(x["pred"]), np.array(x["gt"])
    e = p[:, 3:] - g[:, 3:]
    r = float(np.hypot(*g[0, 3:5]))
    if "park" in (x["state"] or "").lower():
        b = p[0, 3:] - np.median(g[:, 3:], 0)
        park_bias.append(b)
        park_rows.append((v, x["seq"], r, b, np.linalg.norm(e, axis=1), iou(p, g)))
    else:
        u = g[:, 3:5] / np.linalg.norm(g[:, 3:5], axis=1, keepdims=True)
        rad = (e[:, :2] * u).sum(1)
        lat = -e[:, 0] * u[:, 1] + e[:, 1] * u[:, 0]
        mov.append((v, x["seq"], r, rad, lat, e[:, 2], np.linalg.norm(e, axis=1), iou(p, g)))

B = np.array(park_bias)
E = np.concatenate([r[4] for r in park_rows])
print(f"parked correct clips {len(B)}: per-clip bias |dx| |dy| |dz| mean {np.abs(B).mean(0).round(3)} median {np.median(np.abs(B), 0).round(3)}"
      f"  3D mean {np.linalg.norm(B, axis=1).mean():.3f}")
print(f"   per-frame |e|: mean {E.mean():.3f}  <0.1 {np.mean(E < 0.1):.3f}  <0.05 {np.mean(E < 0.05):.3f}"
      f"   SR@0.5 {np.mean(np.concatenate([r[5] for r in park_rows]) > 0.5):.3f}")
for lo, hi in [(0, 20), (20, 30), (30, 60), (60, 90), (90, 300)]:
    rs = [r for r in park_rows if lo <= r[2] < hi]
    if rs:
        bb = np.array([r[3] for r in rs])
        ee = np.concatenate([r[4] for r in rs])
        print(f"   r [{lo},{hi}) n={len(rs):3d} bias mean signed {bb.mean(0).round(3)} |bias| {np.abs(bb).mean(0).round(3)}"
              f"  frame<0.1 {np.mean(ee < 0.1):.2f} <0.05 {np.mean(ee < 0.05):.2f}")
R = np.concatenate([m[3] for m in mov])
L = np.concatenate([m[4] for m in mov])
Z = np.concatenate([m[5] for m in mov])
E = np.concatenate([m[6] for m in mov])
print(f"moving correct clips {len(mov)}: |radial| mean {np.abs(R).mean():.3f} median {np.median(np.abs(R)):.3f}  "
      f"|lateral| {np.abs(L).mean():.3f}  |dz| {np.abs(Z).mean():.3f}   |e|<0.1 {np.mean(E < 0.1):.3f} <0.05 {np.mean(E < 0.05):.3f}"
      f"   radial<0.1 {np.mean(np.abs(R) < 0.1):.3f}")
for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 300)]:
    ms = [m for m in mov if lo <= m[2] < hi]
    if ms:
        print(f"   r [{lo},{hi}) n={len(ms):3d} |rad| {np.abs(np.concatenate([m[3] for m in ms])).mean():.3f} "
              f"|lat| {np.abs(np.concatenate([m[4] for m in ms])).mean():.3f} |dz| {np.abs(np.concatenate([m[5] for m in ms])).mean():.3f}")
