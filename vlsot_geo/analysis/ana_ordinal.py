"""调参“名次”参考框集合：在 TRAIN 真值目标上，比较不同 置信度/最小框高/最大平面距离 过滤下，
按生成脚本定义算出的名次与文本名次的一致率（按目标距离分段）。只用训练标签找真值框。"""
import math
import sys

sys.argv = ["pipeline3.py", "--mode", "cv", "--ground", "global"]
import numpy as np  # noqa: E402

import pipeline3 as P  # noqa: E402

train = P.load_clips("train")
planes, _ = P.fit_ground(train)
offset = P.fit_offset(train)
g = planes["_global"]
cases = []  # (dist, text_ord, target_box, [(box, r_plane), ...])
for c in train:
    ch = P.parse_chunks(c["desc"])
    t = P.parse_text(c["desc"])
    rc = P.range_curve(t["distances"])
    for k, fi in enumerate((0, 14, 29)):
        if ch[k]["ordinal"] is None or rc is None:
            continue
        b = P.gt_box(c, fi)
        if b is None:
            continue
        f = c["frames"][fi]
        calib = P.load_calib(f["img"])
        boxes = P.DETS.get(f"train/{f['img']}", [])
        refs = []
        for r in boxes:
            o, d = P.ray_center(r, calib, offset)
            p = P.on_plane(o, d, g, 0.75)
            refs.append((r, 1e4 if p is None else math.hypot(p[0], p[1])))
        cases.append((rc[fi], ch[k]["ordinal"][0], b, refs))
print("anchor cases with ordinal:", len(cases))


def rank_of(target, refs):
    """按生成脚本的名次定义计算 target 在 refs 中的名次（1 起）：同一九宫格内，按“格子左上角 → 框底边中点”的像素距离排序；
    与 target 的 IoU ≥ 0.7 的框视为同一辆车，不计入。"""
    cell = P.gen_cell(target)
    cx = (target[0] + target[2]) / 2
    cy = (target[1] + target[3]) / 2
    corner = (640 * math.floor(cx / 640), 360 * math.floor(cy / 360))
    d_self = math.hypot(cx - corner[0], target[3] - corner[1])
    return 1 + sum(1 for r in refs if P.gen_cell(r) == cell and P.iou2d(r, target) < 0.7 and
                   math.hypot((r[0] + r[2]) / 2 - corner[0], r[3] - corner[1]) < d_self)


bins = [(0, 30), (30, 60), (60, 90), (90, 1000)]
for conf in (0.2, 0.3, 0.5):
    for hmin in (0, 15, 25):
        for rmax in (1e9, 150, 120):
            agree = {bn: [0, 0] for bn in bins}
            for dist, want, tb, refs in cases:
                ref = [r for r, rp in refs if int(r[5]) in (2, 5, 7) and r[4] >= conf and r[3] - r[1] >= hmin and rp <= rmax]
                ref = [ref[i] for i in P.nms_agnostic(ref)]
                got = rank_of(tb, ref)
                for bn in bins:
                    if bn[0] <= dist < bn[1]:
                        agree[bn][0] += got == want
                        agree[bn][1] += 1
            tot = sum(a for a, _ in agree.values()) / max(1, sum(n for _, n in agree.values()))
            print(f"conf {conf} hmin {hmin:2d} rmax {rmax:>9.0f}: all {tot:.3f} | " +
                  " ".join(f"{bn[0]}-{bn[1]}m {a / max(n, 1):.2f}(n={n})" for bn, (a, n) in agree.items()))
