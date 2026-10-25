"""扁/窄框的尺寸放大离线检验（交叉验证 dump，只用 TRAIN 标签评估）：
把小于阈值 T 的那一维乘以 s，看全体与“含扁框片段”的 SR@0.5 / SR@0.7 / AOR 变化（中心不变，ACE/PR 不变）。
用法：python ana_thin.py "runs/ablation/a9_v12f_f?.json" """
import glob
import json
import sys

import numpy as np

from geo_common import evaluate

d = {}
for p in sorted(glob.glob(sys.argv[1])):
    d.update(json.load(open(p)))
pred = {v: np.array(x["pred"]) for v, x in d.items()}
gt = {v: np.array(x["gt"]) for v, x in d.items()}
thin = [v for v in d if (pred[v][0, :3] < 1.0).any()]
print("clips with a dim < 1.0 m:", len(thin))
b_all, b_thin = evaluate(pred, gt), evaluate({v: pred[v] for v in thin}, {v: gt[v] for v in thin})
print("base   ALL SR %.2f SR7 %.2f AOR %.2f | thin SR %.2f SR7 %.2f AOR %.2f" % (
    b_all["SR@0.5"], b_all["SR@0.7"], b_all["AOR"], b_thin["SR@0.5"], b_thin["SR@0.7"], b_thin["AOR"]))
for T in (0.8, 1.0, 1.2):
    for s in (1.05, 1.1, 1.15, 1.2):
        q = {}
        for v, p in pred.items():
            p2 = p.copy()
            m = p2[:, :3] < T
            p2[:, :3] = np.where(m, p2[:, :3] * s, p2[:, :3])
            q[v] = p2
        a = evaluate(q, gt)
        t = evaluate({v: q[v] for v in thin}, {v: gt[v] for v in thin})
        print("T %.1f s %.2f ALL SR %.2f SR7 %.2f AOR %.2f | thin SR %.2f SR7 %.2f AOR %.2f" % (
            T, s, a["SR@0.5"], a["SR@0.7"], a["AOR"], t["SR@0.5"], t["SR@0.7"], t["AOR"]))
