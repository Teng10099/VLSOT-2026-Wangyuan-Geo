"""对比两个测试输出目录（只看我们自己的预测）：逐片段 选中框是否相同、xyz 平均差。
用法：python cmp_runs.py out/test_A out/test_B"""
import json
import sys

import numpy as np

a, b = sys.argv[1], sys.argv[2]
ta, tb = json.load(open(f"{a}/tracks.json")), json.load(open(f"{b}/tracks.json"))
for vid in sorted(ta):
    pa, pb = np.loadtxt(f"{a}/result/{vid}.txt"), np.loadtxt(f"{b}/result/{vid}.txt")
    same = np.mean([x is not None and y is not None and list(x) == list(y) for x, y in zip(ta[vid], tb[vid])])
    d = pa[:, 4:7] - pb[:, 4:7]
    print(f"{vid}: same box {same:.2f}  mean dxyz {d.mean(0).round(3)}  mean |d| {np.linalg.norm(d, axis=1).mean():.3f}"
          f"  A f0 {pa[0, 4:7].round(2)} f29 {pa[29, 4:7].round(2)}  B f0 {pb[0, 4:7].round(2)}")
