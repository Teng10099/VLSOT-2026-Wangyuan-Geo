"""中心残差回归（两层交叉验证，只用 TRAIN 标签）：用 v12f 交叉验证的“折外”预测，
学习 GT 中心 - 预测中心 在 横向 / 高度 上的残差（特征：检测框几何、截断、文本尺寸、距离、航向与视线夹角、运动与否、类型），
再对另一折的预测做修正。第 k 折的修正模型只用其它折的片段训练。
用法：python ana_resid.py "runs/ablation/a9_v12f_f?.json" """
import glob
import json
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from geo_common import evaluate, load_clips, parse_text

files = sorted(glob.glob(sys.argv[1]))
fold_of, d = {}, {}
for k, p in enumerate(files):
    for v, x in json.load(open(p)).items():
        d[v], fold_of[v] = x, k
clips = {c["vid"]: c for c in load_clips("train")}
TYPES = ["Car", "Van", "Truck", "Bus"]


def feats(x, desc, typ):
    """一段 dump 的逐帧残差回归特征 (30, 19) 与有效帧掩码：检测框中心与宽高（归一化）、log 宽高比、四边是否截断、
    文本尺寸 l w h、log 距离、航向与视线夹角的 sin / cos、是否运动、车辆类型（独热）。没有选中框的帧特征为 0、掩码为 False。"""
    t = parse_text(desc)
    az = np.radians(t["azimuth"] or 0.0)
    p = np.array(x["pred"])
    F, keep = [], []
    for fi in range(30):
        b = x["sel"][fi]
        if b is None:
            F.append(np.zeros(19))
            keep.append(False)
            continue
        cx, cy, bw, bh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
        r = np.hypot(p[fi, 3], p[fi, 4])
        bear = np.arctan2(p[fi, 4], p[fi, 3])
        F.append(np.r_[cx / 1920, cy / 1080, bw / 1920, bh / 1080, np.log(max(bh, 1) / max(bw, 1)),
                       [b[0] < 10, b[1] < 10, b[2] > 1910, b[3] > 1070], p[fi, :3], np.log(r),
                       np.sin(az - bear), np.cos(az - bear), "mov" in (x["state"] or "").lower(),
                       [typ == tt for tt in TYPES[:3]]])
        keep.append(True)
    return np.array(F, dtype=np.float64), np.array(keep)


rows = {}
for v, x in d.items():
    c = clips[v]
    F, keep = feats(x, c["desc"], c["type"])
    p, g = np.array(x["pred"]), np.array(x["gt"])
    u = p[:, 3:5] / np.linalg.norm(p[:, 3:5], axis=1, keepdims=True)
    lat_dir = np.stack([-u[:, 1], u[:, 0]], 1)
    e = g[:, 3:] - p[:, 3:]
    y = np.stack([(e[:, :2] * lat_dir).sum(1), e[:, 2]], 1)
    ok = np.array([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])])
    rows[v] = (F, y, keep & ok, lat_dir)

gts = {v: np.array(x["gt"]) for v, x in d.items()}
base = {v: np.array(x["pred"]) for v, x in d.items()}
models = {"ridge": lambda: Ridge(alpha=10.0),
          "gbr": lambda: GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05,
                                                   subsample=0.8, min_samples_leaf=150)}
for mname, make in models.items():
    for shrink in (0.5, 1.0):
        out = {}
        for k in range(len(files)):
            tr = [v for v in d if fold_of[v] != k]
            X = np.concatenate([rows[v][0][rows[v][2]] for v in tr])
            Y = np.concatenate([rows[v][1][rows[v][2]] for v in tr])
            ms = [make().fit(X, Y[:, j]) for j in range(2)]
            for v in d:
                if fold_of[v] != k:
                    continue
                F, _, keep, lat_dir = rows[v]
                corr = np.stack([m.predict(F) for m in ms], 1) * shrink
                corr[~keep] = 0
                p = base[v].copy()
                if "park" in (d[v]["state"] or "").lower():
                    corr[:] = np.median(corr[keep], 0) if keep.any() else 0
                p[:, 3:5] += corr[:, :1] * lat_dir
                p[:, 5] += corr[:, 1] if "park" not in (d[v]["state"] or "").lower() else np.median(corr[:, 1])
                out[v] = p
        e0, e1 = evaluate(base, gts), evaluate(out, gts)
        print(f"{mname:5s} shrink {shrink}: " + "  ".join(f"{key} {e0[key]:.2f}->{e1[key]:.2f}" for key in
                                                        ("SR@0.5", "SR@0.7", "AOR", "ACE", "PR10cm", "PR5cm")))
