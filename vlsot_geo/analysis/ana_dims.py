"""文本尺寸 vs 逐帧标签尺寸（只用 TRAIN 标签）：文本是否等于第 0 帧尺寸？其余帧偏离多少？
有没有比“整段都用文本尺寸”更好的逐帧尺寸估计（例如按类别线性收缩）？"""
import collections

import numpy as np

from geo_common import load_clips, parse_text

clips = [c for c in load_clips("train") if parse_text(c["desc"])["lwh"] is not None]
print("clips with text dims:", len(clips))
d0, dev_t, rows = [], collections.defaultdict(list), []
for c in clips:
    t = parse_text(c["desc"])["lwh"]
    lab = np.array([f["lwh"] for f in c["frames"]])
    d0.append(np.abs(lab[0] - t))
    for fi in range(30):
        dev_t[fi].append(np.abs(lab[fi] - t))
    rows.append((c["type"], t, lab))
d0 = np.array(d0)
print("|frame0 label - text| mean", d0.mean(0).round(3), "max", d0.max(0).round(3), "frac<=0.051", (d0 <= 0.051).mean(0).round(3))
for fi in (0, 1, 5, 10, 14, 20, 29):
    print(f"  frame {fi:2d}: mean |label - text| l/w/h = {np.mean(dev_t[fi], 0).round(3)}")
allerr = np.concatenate([np.abs(lab - t) for _, t, lab in rows])
print("all frames mean |label - text|:", allerr.mean(0).round(3))
# 片段内尺寸是否恒定
within = np.concatenate([np.abs(lab - np.median(lab, 0)) for _, t, lab in rows])
print("all frames mean |label - clip median| (oracle per-clip constant):", within.mean(0).round(3))
# 按类别的线性收缩：label_t ≈ a + b*text，留一片段交叉验证
types = sorted({r[0] for r in rows})
pred_err = []
for i, (tp, t, lab) in enumerate(rows):
    tr = [(t2, l2) for j, (tp2, t2, l2) in enumerate(rows) if j != i and tp2 == tp]
    if len(tr) < 5:
        tr = [(t2, l2) for j, (tp2, t2, l2) in enumerate(rows) if j != i]
    X = np.concatenate([np.repeat(t2[None], 30, 0) for t2, _ in tr])
    Y = np.concatenate([l2 for _, l2 in tr])
    p = np.zeros(3)
    for k in range(3):
        A = np.c_[np.ones(len(X)), X[:, k]]
        coef = np.linalg.lstsq(A, Y[:, k], rcond=None)[0]
        p[k] = coef[0] + coef[1] * t[k]
    pred_err.append(np.abs(lab - p))
print("LOO linear shrink per type  mean |label - pred|:", np.concatenate(pred_err).mean(0).round(3))
# 只在 frame>0 用收缩、frame 0 用文本
mix = []
for (tp, t, lab), e in zip(rows, pred_err):
    m = e.copy()
    m[0] = np.abs(lab[0] - t)
    mix.append(m)
print("text at f0 + shrink elsewhere:", np.concatenate(mix).mean(0).round(3))
for tp in types:
    sub = [(t, lab) for tp2, t, lab in rows if tp2 == tp]
    e = np.concatenate([np.abs(lab - t) for t, lab in sub])
    print(f"  type {tp:6s} clips {len(sub):3d}  mean |label - text| {e.mean(0).round(3)}  text h mean {np.mean([t[2] for t, _ in sub]):.2f}"
          f"  label h mean {np.mean([lab[:, 2].mean() for _, lab in sub]):.2f}")
