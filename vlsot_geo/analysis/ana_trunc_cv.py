"""比较两组交叉验证 dump（A=ray, B=boxfit）在截断片段上的逐段表现（只用 TRAIN 标签）：
截断边（左/上/右/下，10px）、距离、IoU、SR、有符号误差 dx dy dz。
用法：python ana_trunc_cv.py "runs/ablation/a5_lin_f?.json" "runs/ablation/a8_bfall_f?.json" """
import glob
import json
import sys

import numpy as np


def load(pat):
    """读取并合并匹配 glob 模式的各折交叉验证 dump。"""
    d = {}
    for p in sorted(glob.glob(pat)):
        d.update(json.load(open(p)))
    return d


def iou(p, g):
    """逐帧的正常 3D IoU：p、g 为 (30, 6) 的 l w h x y z 数组。"""
    lo_p, hi_p = p[:, 3:] - p[:, :3] / 2, p[:, 3:] + p[:, :3] / 2
    lo_g, hi_g = g[:, 3:] - g[:, :3] / 2, g[:, 3:] + g[:, :3] / 2
    inter = np.clip(np.minimum(hi_p, hi_g) - np.maximum(lo_p, lo_g), 0, None).prod(1)
    return inter / (p[:, :3].prod(1) + g[:, :3].prod(1) - inter)


A, B = load(sys.argv[1]), load(sys.argv[2])
m = 10
print("vid    seq  r     sides(LTRB frac)        | A: SR  IoU  dxyz            | B: SR  IoU  dxyz")
tot = []
for v in sorted(A):
    x = A[v]
    boxes = [b for b in x["sel"] if b is not None]
    if not boxes:
        continue
    fr = np.mean([[b[0] < m, b[1] < m, b[2] > 1920 - m, b[3] > 1080 - m] for b in boxes], 0)
    if fr.max() < 0.3:
        continue
    g = np.array(x["gt"])
    pa, pb = np.array(x["pred"]), np.array(B[v]["pred"])
    ia, ib = iou(pa, g), iou(pb, g)
    r = np.hypot(*g[0, 3:5])
    tot.append((r, (ia > 0.5).mean(), (ib > 0.5).mean()))
    print(f"{v} {x['seq']} {r:5.1f}  {np.round(fr, 1)} | {100 * (ia > 0.5).mean():5.1f} {ia.mean():.2f} {(pa - g)[:, 3:].mean(0).round(2)} "
          f"| {100 * (ib > 0.5).mean():5.1f} {ib.mean():.2f} {(pb - g)[:, 3:].mean(0).round(2)}")
t = np.array(tot)
for lo, hi in [(0, 20), (20, 40), (40, 300)]:
    s = (t[:, 0] >= lo) & (t[:, 0] < hi)
    if s.any():
        print(f"r [{lo},{hi}) clips {s.sum()}: A SR {100 * t[s, 1].mean():.1f}  B SR {100 * t[s, 2].mean():.1f}")
