"""运动目标：用选中检测框底边中点射线与全局地平面的交点距离 r_plane(t) 作为“相对距离变化”的线索，
修正文本距离插值：r(t) = r_quad(t) + w * (r_plane_s(t) - quad_through_keyframes(r_plane_s)(t))。
在交叉验证 dump 上离线评估径向误差（只用 TRAIN 标签评估）。用法：python ana_rplane.py "runs/ablation/a9_v12f_f?.json" """
import glob
import json
import sys

import numpy as np

from geo_common import load_calib, load_clips, parse_text, pixel_rays

d = {}
for p in sorted(glob.glob(sys.argv[1])):
    d.update(json.load(open(p)))
clips = {c["vid"]: c for c in load_clips("train")}
rows = np.array([[f["xyz"][0], f["xyz"][1], f["xyz"][2] - f["lwh"][2] / 2] for c in clips.values() for f in c["frames"]])
plane = np.linalg.lstsq(np.c_[rows[:, 0], rows[:, 1], np.ones(len(rows))], rows[:, 2], rcond=None)[0]


def local_linear(v, k):
    """逐点局部线性回归平滑，窗口 ±k（同 pipeline3.local_linear）。"""
    ts = np.arange(len(v))
    out = np.empty(len(v))
    for i in range(len(v)):
        lo, hi = max(0, i - k), min(len(v), i + k + 1)
        A = np.c_[ts[lo:hi] - i, np.ones(hi - lo)]
        out[i] = np.linalg.lstsq(A, v[lo:hi], rcond=None)[0][1]
    return out


def ground_range(box, img):
    """框底边中点的视线与全局地平面交点的水平距离（米）。"""
    o, dirs = pixel_rays([[(box[0] + box[2]) / 2, box[3]]], load_calib(img))
    dd = dirs[0]
    a, b, c = plane
    n = np.array([a, b, -1.0])
    s = -(o @ n + c) / (dd @ n)
    p = o + s * dd
    return float(np.hypot(p[0], p[1]))


res = {k: [] for k in ("quad", "w0.5_k3", "w1_k3", "w1_k5", "w0.5_k5")}
for v, x in d.items():
    if "mov" not in (x["state"] or "").lower():
        continue
    sel = np.mean([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])])
    if sel < 0.8 or any(s is None for s in x["sel"]):
        continue
    t = parse_text(clips[v]["desc"])
    if len(t["distances"]) < 3:
        continue
    g = np.array(x["gt"])
    rg = np.hypot(g[:, 3], g[:, 4])
    ts = np.arange(30)
    quad = np.polyval(np.polyfit([0, 14, 29], t["distances"][:3], 2), ts)
    rp = np.array([ground_range(b, img) for b, img in zip(x["sel"], x["imgs"])])
    res["quad"].append(quad - rg)
    for k in (3, 5):
        rps = local_linear(rp, k)
        base = np.polyval(np.polyfit([0, 14, 29], rps[[0, 14, 29]], 2), ts)
        for wgt in (0.5, 1.0):
            key = f"w{wgt:g}_k{k}"
            if key in res:
                res[key].append(quad + wgt * (rps - base) - rg)
for k, v in res.items():
    a = np.abs(np.concatenate(v))
    print(f"{k:8s} clips {len(v)}  |radial| mean {a.mean():.3f} median {np.median(a):.3f}  <0.1 {np.mean(a < 0.1):.3f}  <0.05 {np.mean(a < 0.05):.3f}")
