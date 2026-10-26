"""运动目标的误差随帧号的分布（交叉验证 dump，只看选对的片段）：
若误差在关键帧 0/14/29 处小、中间大，说明是“距离线性插值”带来的误差。
另外分解为 径向误差（沿视线距离）与 横向误差。用法：python ana_move.py runs/ablation/a3_v11c_f?.json"""
import json
import sys

import numpy as np


def iou(p, g):
    """逐帧的正常 3D IoU：p、g 为 (30, 6) 的 l w h x y z 数组。"""
    lo_p, hi_p = p[:, 3:] - p[:, :3] / 2, p[:, 3:] + p[:, :3] / 2
    lo_g, hi_g = g[:, 3:] - g[:, :3] / 2, g[:, 3:] + g[:, :3] / 2
    inter = np.clip(np.minimum(hi_p, hi_g) - np.maximum(lo_p, lo_g), 0, None).prod(1)
    return inter / (p[:, :3].prod(1) + g[:, :3].prod(1) - inter)


d = {}
for path in sys.argv[1:]:
    d.update(json.load(open(path)))
E, RAD, LAT, I = [], [], [], []
for v, x in d.items():
    if "mov" not in (x["state"] or "").lower():
        continue
    sel = np.mean([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])])
    if sel < 0.8:
        continue
    p, g = np.array(x["pred"]), np.array(x["gt"])
    e = p[:, 3:] - g[:, 3:]
    u = g[:, 3:5] / np.linalg.norm(g[:, 3:5], axis=1, keepdims=True)
    rad = (e[:, :2] * u).sum(1)
    lat = e[:, 0] * -u[:, 1] + e[:, 1] * u[:, 0]
    E.append(np.linalg.norm(e, axis=1))
    RAD.append(rad)
    LAT.append(lat)
    I.append(iou(p, g))
E, RAD, LAT, I = map(np.array, (E, RAD, LAT, I))
print("moving correct clips", len(E))
print("frame  |e|   |radial| |lateral|  SR@0.5")
for fi in range(30):
    print(f"  {fi:2d}  {E[:, fi].mean():.3f}  {np.abs(RAD[:, fi]).mean():.3f}   {np.abs(LAT[:, fi]).mean():.3f}   {100 * (I[:, fi] > 0.5).mean():5.1f}")
print("overall |radial| %.3f |lateral| %.3f" % (np.abs(RAD).mean(), np.abs(LAT).mean()))
