"""在交叉验证 dump 上离线检验 traj_fit（只用 TRAIN 标签评估）：运动目标的 x,y 用轨迹拟合替换，z 不变。
用法：python ana_traj.py "runs/ablation/a9_v12f_f?.json" """
import glob
import json
import sys

import numpy as np

from geo_common import evaluate, load_calib, load_clips, parse_text, pixel_rays
from traj_fit import fit_traj

d = {}
for p in sorted(glob.glob(sys.argv[1])):
    d.update(json.load(open(p)))
clips = {c["vid"]: c for c in load_clips("train")}


def cam_xy(img):
    o, _ = pixel_rays([[960.0, 540.0]], load_calib(img))
    return o[:2]


variants = {"deg1": dict(deg=1), "deg2": dict(deg=2), "deg2_acc1": dict(deg=2, sig_acc=1.0),
            "deg3": dict(deg=3), "deg2_lat2x": dict(deg=2, lat_scale=2.0)}
preds = {k: {} for k in ["orig"] + list(variants)}
gts, movs = {}, []
for v, x in d.items():
    p, g = np.array(x["pred"]), np.array(x["gt"])
    gts[v] = g
    for k in preds:
        preds[k][v] = p.copy()
    if "mov" in (x["state"] or "").lower():
        t = parse_text(clips[v]["desc"])
        if len(t["distances"]) < 3:
            continue
        movs.append(v)
        cams = np.array([cam_xy(img) for img in x["imgs"]])
        for k, kw in variants.items():
            kw = dict(kw)
            ls = kw.pop("lat_scale", 1.0)
            dist = np.linalg.norm(p[:, 3:5] - cams, axis=1)
            xy = fit_traj(p[:, 3:5], cams, [0, 14, 29], t["distances"][:3], sig_lat=ls * (0.03 + 0.0015 * dist), **kw)
            preds[k][v][:, 3:5] = xy
for k, pr in preds.items():
    a = evaluate(pr, gts)
    m = evaluate({v: pr[v] for v in movs}, {v: gts[v] for v in movs})
    print(f"{k:12s} ALL SR {a['SR@0.5']:.2f} SR7 {a['SR@0.7']:.2f} AOR {a['AOR']:.2f} ACE {a['ACE']:.3f} PR10 {a['PR10cm']:.2f} PR5 {a['PR5cm']:.2f} | "
          f"moving({len(movs)}) SR {m['SR@0.5']:.2f} SR7 {m['SR@0.7']:.2f} AOR {m['AOR']:.2f} ACE {m['ACE']:.3f} PR10 {m['PR10cm']:.2f} PR5 {m['PR5cm']:.2f}")
