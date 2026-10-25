"""运动目标：文本距离（第 0/14/29 帧，0.1 m 取整）插值成逐帧距离的误差（只用 TRAIN 标签）。
比较 分段线性 / 过三点的二次曲线 两种插值，与 GT 水平距离 hypot(x, y) 的偏差。"""
import numpy as np

from geo_common import load_clips, parse_text

clips = load_clips("train")
lin_e, quad_e, n = [], [], 0
per_frame_lin, per_frame_quad = np.zeros(30), np.zeros(30)
for c in clips:
    if "mov" not in (c["state"] or "").lower():
        continue
    t = parse_text(c["desc"])
    if len(t["distances"]) < 3 or any(f["xyz"] is None for f in c["frames"]):
        continue
    g = np.array([np.hypot(f["xyz"][0], f["xyz"][1]) for f in c["frames"]])
    g3 = np.array([np.linalg.norm(f["xyz"]) for f in c["frames"]])
    d = np.array(t["distances"][:3])
    if np.abs(d - g3[[0, 14, 29]]).max() > 0.3 and np.abs(d - g[[0, 14, 29]]).max() > 0.3:
        continue  # 文本距离与 GT 对不上（解析错误），跳过
    lin = np.interp(np.arange(30), [0, 14, 29], d)
    quad = np.polyval(np.polyfit([0, 14, 29], d, 2), np.arange(30))
    lin_e.append(np.abs(lin - g))
    quad_e.append(np.abs(quad - g))
    n += 1
lin_e, quad_e = np.array(lin_e), np.array(quad_e)
print("moving clips", n)
print("mean |r_interp - r_gt|: linear %.3f  quadratic %.3f" % (lin_e.mean(), quad_e.mean()))
print("p90: linear %.3f quadratic %.3f" % (np.percentile(lin_e, 90), np.percentile(quad_e, 90)))
print("frame  linear  quadratic")
for fi in range(0, 30, 2):
    print(f"  {fi:2d}   {lin_e[:, fi].mean():.3f}   {quad_e[:, fi].mean():.3f}")
d3 = []
for c in clips:
    t = parse_text(c["desc"])
    if len(t["distances"]) >= 3 and all(c["frames"][k]["xyz"] is not None for k in (0, 14, 29)):
        for j, k in enumerate((0, 14, 29)):
            x = c["frames"][k]["xyz"]
            d3.append((t["distances"][j] - np.hypot(x[0], x[1]), t["distances"][j] - np.linalg.norm(x)))
d3 = np.array(d3)
print("text - hypot(x,y): mean %.3f MAD %.3f | text - |xyz|: mean %.3f MAD %.3f" % (
    np.median(d3[:, 0]), np.median(np.abs(d3[:, 0])), np.median(d3[:, 1]), np.median(np.abs(d3[:, 1]))))
