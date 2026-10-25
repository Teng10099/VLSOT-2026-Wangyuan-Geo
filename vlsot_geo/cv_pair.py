"""两组交叉验证 dump 的逐段配对比较（正常 3D IoU 口径）：变好/变差的片段数、按折的 SR、差值的 bootstrap 区间。
用法：python cv_pair.py "runs/ablation/a3_v11c_f?.json" "runs/ablation/a8_bfall_f?.json" """
import glob
import json
import sys

import numpy as np


def load(pat):
    out = {}
    for k, p in enumerate(sorted(glob.glob(pat))):
        for v, x in json.load(open(p)).items():
            x["fold"] = k
            out[v] = x
    return out


def sr(x):
    p, g = np.array(x["pred"]), np.array(x["gt"])
    lo = np.maximum(p[:, 3:] - p[:, :3] / 2, g[:, 3:] - g[:, :3] / 2)
    hi = np.minimum(p[:, 3:] + p[:, :3] / 2, g[:, 3:] + g[:, :3] / 2)
    inter = np.clip(hi - lo, 0, None).prod(1)
    iou = inter / (p[:, :3].prod(1) + g[:, :3].prod(1) - inter)
    return 100 * (iou > 0.5).mean(), iou.mean()


A, B = load(sys.argv[1]), load(sys.argv[2])
vids = sorted(set(A) & set(B))
a = np.array([sr(A[v])[0] for v in vids])
b = np.array([sr(B[v])[0] for v in vids])
d = b - a
print(f"clips {len(vids)}: A {a.mean():.2f}  B {b.mean():.2f}  diff {d.mean():+.2f}")
print(f"  better (>+5) {np.sum(d > 5)}  worse (<-5) {np.sum(d < -5)}  same {np.sum(np.abs(d) <= 5)}")
folds = np.array([A[v]["fold"] for v in vids])
print("  per fold diff:", [round(float(d[folds == k].mean()), 2) for k in range(5)])
rng = np.random.RandomState(0)
bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
print(f"  bootstrap 95% CI of diff: [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
worst = np.argsort(d)[:8]
print("  most worsened:", [(vids[i], A[vids[i]]["seq"], round(float(a[i]), 1), round(float(b[i]), 1)) for i in worst])
best = np.argsort(-d)[:8]
print("  most improved:", [(vids[i], A[vids[i]]["seq"], round(float(a[i]), 1), round(float(b[i]), 1)) for i in best])
