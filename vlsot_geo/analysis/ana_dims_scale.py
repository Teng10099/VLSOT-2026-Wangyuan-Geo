"""尺寸缩放离线检验（交叉验证 dump，只用 TRAIN 标签评估）：
GT 每帧尺寸与文本尺寸（第 0 帧、0.1 m 取整）不同；检查 l/w/h 各乘一个系数后 SR@0.5 / SR@0.7 / AOR 的变化，
以及 GT 尺寸相对文本尺寸的有符号偏差。中心不变，所以 ACE / PR 不受影响。
用法：python ana_dims_scale.py "runs/ablation/a9_v12f_f?.json" """
import glob
import itertools
import json
import sys

import numpy as np

from geo_common import evaluate

d = {}
for p in sorted(glob.glob(sys.argv[1])):
    d.update(json.load(open(p)))
pred = {v: np.array(x["pred"]) for v, x in d.items()}
gt = {v: np.array(x["gt"]) for v, x in d.items()}
dl = np.concatenate([gt[v][:, :3] - pred[v][:, :3] for v in d])
print("GT dims - text dims: mean", dl.mean(0).round(3), " median", np.median(dl, 0).round(3),
      " ratio median", np.median(np.concatenate([gt[v][:, :3] / pred[v][:, :3] for v in d]), 0).round(3))
mov = [v for v in d if "mov" in (d[v]["state"] or "").lower()]
park = [v for v in d if v not in set(mov)]
for name, vs in (("parked", park), ("moving", mov)):
    x = np.concatenate([gt[v][:, :3] / pred[v][:, :3] for v in vs])
    print(f"  {name}: GT/text ratio median {np.median(x, 0).round(3)}  mean {x.mean(0).round(3)}")
base = evaluate(pred, gt)
print("base", {k: base[k] for k in ("SR@0.5", "SR@0.7", "AOR")})
rows = []
for sl, sw, sh in itertools.product([0.94, 0.97, 1.0, 1.03], [0.94, 0.97, 1.0, 1.03, 1.06], [0.94, 0.97, 1.0, 1.03, 1.06]):
    q = {}
    for v, p in pred.items():
        p2 = p.copy()
        p2[:, :3] *= [sl, sw, sh]
        q[v] = p2
    e = evaluate(q, gt)
    rows.append((sl, sw, sh, e["SR@0.5"], e["SR@0.7"], e["AOR"]))
rows.sort(key=lambda r: -(r[3] + r[4] + r[5]))
print("best by SR@0.5+SR@0.7+AOR:")
for r in rows[:8]:
    print("  scale l %.2f w %.2f h %.2f -> SR@0.5 %.2f SR@0.7 %.2f AOR %.2f" % r)
