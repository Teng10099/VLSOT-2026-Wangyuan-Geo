"""EFN 颜色概率与文本颜色在 TRAIN 真值框上的一致程度（只用训练标签找真值框）。"""
import sys

sys.argv = ["pipeline3.py", "--mode", "cv", "--color_model", "efn"]
import numpy as np  # noqa: E402

import pipeline3 as P  # noqa: E402

rows = []
for c in P.load_clips("train"):
    name = P.color_of(c)
    if name not in P.EFN_GROUPS:
        continue
    ps = []
    for fi in range(30):
        key = f"train/{c['frames'][fi]['img']}"
        probs = P.EFN["probs"].get(key)
        b = P.gt_box(c, fi)
        if probs is None or b is None:
            continue
        idx = [i for i, bb in enumerate(P.DETS[key]) if list(bb) == list(b)]
        if not idx or not np.isfinite(probs[idx[0]]).all():
            continue
        ps.append(sum(probs[idx[0]][P.EFN_INDEX[g]] for g in P.EFN_GROUPS[name]))
    if ps:
        rows.append((c["vid"], name, float(np.mean(ps)), float(np.median(ps)), len(ps)))
m = np.array([r[2] for r in rows])
print("clips", len(rows), "mean p(text colour) on GT boxes: mean %.2f median %.2f" % (m.mean(), np.median(m)))
for t in (0.02, 0.05, 0.1, 0.2, 0.5):
    print(f"  fraction of clips with clip-mean p < {t}: {np.mean(m < t):.3f}")
for name in sorted({r[1] for r in rows}):
    s = [r[2] for r in rows if r[1] == name]
    print(f"  {name:14s} n={len(s):3d} mean p {np.mean(s):.2f}  frac<0.1 {np.mean(np.array(s) < 0.1):.2f}")
print("lowest:", sorted(rows, key=lambda r: r[2])[:8])
