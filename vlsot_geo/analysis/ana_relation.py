"""检查“与最近车方位关系 / 最近车颜色”线索在 TRAIN 上是否可靠（只用训练标签找真值框）。
对每段关键帧：取真值目标对应的检测框，按与 pipeline3 相同的方式找最近车辆，比较算出的方位与文本方位、
最近车 CLIP 颜色与文本颜色。"""
import collections
import sys

sys.argv = ["pipeline3.py", "--mode", "cv", "--rel_feat", "1", "--near_color_feat", "1", "--color_model", "clip",
            "--ordinal_mode", "gen", "--ground", "global"]
import numpy as np  # noqa: E402

import pipeline3 as P  # noqa: E402

train = P.load_clips("train")
planes, _ = P.fit_ground(train)
offset = P.fit_offset(train)
rel_ok, rel_n, nc_ok, nc_n = 0, 0, 0, 0
conf = collections.Counter()
ord_ok, ord_n = 0, 0
for c in train:
    cc = P.clip_candidates(c, planes, offset, None)
    ch = P.parse_chunks(c["desc"])
    for k, fi in enumerate((0, 14, 29)):
        b = P.gt_box(c, fi)
        if b is None:
            continue
        hit = [cd for cd in cc["frames"][fi] if list(cd["box"]) == list(b)]
        if not hit:
            continue
        cd = hit[0]
        if ch[k]["rel"] is not None and "rel" in cd:
            rel_n += 1
            rel_ok += cd["rel"] == ch[k]["rel"]
            conf[(ch[k]["rel"], cd["rel"])] += 1
        if ch[k]["near_color"] is not None and cd["feat"][10] > 0:
            nc_n += 1
            nc_ok += cd["feat"][10] < 1.0  # p > 0.37
        if ch[k]["ordinal"] is not None and "ord" in cd:
            ord_n += 1
            ord_ok += cd["ord"] == ch[k]["ordinal"][0]
print(f"relation agreement {rel_ok}/{rel_n} = {rel_ok / max(rel_n, 1):.2f}")
print(f"near colour p>0.37 {nc_ok}/{nc_n} = {nc_ok / max(nc_n, 1):.2f}")
print(f"ordinal exact {ord_ok}/{ord_n} = {ord_ok / max(ord_n, 1):.2f}")
print("top (text, computed) relation pairs:", conf.most_common(12))
