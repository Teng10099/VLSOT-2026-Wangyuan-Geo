"""汇总 5 折交叉验证的逐段结果：python cv_agg.py name dump_f0.json dump_f1.json ...
主指标为正常 3D IoU 口径（geo_common.evaluate；由 v9→v11c 的平台 ACE 变化推断平台用的就是它），
PR10cm/PR5cm 对应平台 PR@1.0/PR@0.5 列（中心误差 <0.1 m / <0.05 m）。旧的 evaluate_platform 口径只作参考。
test-cams 分组 = 与测试片段同一 sequence（相机）的 TRAIN 片段，sequence 号取自测试 json（公开输入）。"""
import json
import sys

import numpy as np

from geo_common import evaluate, evaluate_platform, load_clips

try:
    TEST_SEQS = {c["seq"] for c in load_clips("test")}
except OSError:
    TEST_SEQS = set()
name = sys.argv[1]
d = {}
for p in sys.argv[2:]:
    d.update(json.load(open(p)))
pred = {v: np.array(x["pred"]) for v, x in d.items()}
gt = {v: np.array(x["gt"]) for v, x in d.items()}


def trunc(x):
    """该片段是否有某一帧的 GT 对应检测框贴图像边界（4 像素以内），用于 trunc 分组。"""
    return any(b is not None and (b[0] < 4 or b[1] < 4 or b[2] > 1916 or b[3] > 1076) for b in x["gtbox"])


groups = {
    "test-cams": [v for v in d if d[v]["seq"] in TEST_SEQS],
    "moving": [v for v in d if "mov" in (d[v]["state"] or "").lower()],
    "parked": [v for v in d if "mov" not in (d[v]["state"] or "").lower()],
    "near<30": [v for v in d if np.hypot(*gt[v][0, 3:5]) < 30],
    "trunc": [v for v in d if trunc(d[v])],
}
tr = evaluate(pred, gt)
pl = evaluate_platform(pred, gt)
sel = np.mean([np.mean([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])])
               for x in d.values()])
print(f"{name:22s} n={len(d)} TRUE SR@0.5 {tr['SR@0.5']:.2f} SR@0.7 {tr['SR@0.7']:.2f} AOR {tr['AOR']:.2f} "
      f"ACE {tr['ACE']:.3f} PR10cm {tr['PR10cm']:.2f} PR5cm {tr['PR5cm']:.2f} | sel {sel:.3f} | old-plat SR {pl['SR@0.5']:.2f}")
line = "    "
for g, vs in groups.items():
    if vs:
        e = evaluate({v: pred[v] for v in vs}, {v: gt[v] for v in vs})
        line += f"{g}({len(vs)}) {e['SR@0.5']:.1f}/{e['AOR']:.1f}  "
print(line)
