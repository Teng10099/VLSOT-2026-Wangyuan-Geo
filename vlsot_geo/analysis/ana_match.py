"""GT ↔ 检测框匹配方式对比（只用 TRAIN 标签）：
center = 含 GT 中心投影的最小框（pipeline3.gt_box 现用）；rect = 与 GT 3D 框投影矩形 IoU 最大的框。
统计两者不一致的比例（按 sequence），以及 center 匹配框与投影矩形的 IoU 分布。"""
import collections
import json

import numpy as np

from boxfit import rects
from geo_common import WORK, load_calib, load_clips, parse_text, project

DETS = json.load(open(WORK + "/cache/dets_yolo11x_1920.json"))


def contains(b, uv):
    return b[0] <= uv[0] <= b[2] and b[1] <= uv[1] <= b[3]


def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


st = collections.defaultdict(lambda: collections.Counter())
low = collections.defaultdict(list)
for c in load_clips("train"):
    t = parse_text(c["desc"])
    if t["lwh"] is None:
        continue
    yaw = np.radians(t["azimuth"] or 0.0)
    for fi in range(0, 30, 2):
        f = c["frames"][fi]
        if f["xyz"] is None:
            continue
        calib = load_calib(f["img"])
        boxes = DETS.get(f"train/{f['img']}", [])
        uv, _ = project(f["xyz"][None], calib)
        hits = [b for b in boxes if contains(b, uv[0])]
        bc = min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if hits else None
        R, ok = rects(f["xyz"][None], t["lwh"], yaw, calib)
        r = [max(0, R[0, 0]), max(0, R[0, 1]), min(1920, R[0, 2]), min(1080, R[0, 3])]
        br = max(boxes, key=lambda b: iou2d(b, r), default=None) if ok[0] and r[2] > r[0] and r[3] > r[1] else None
        s = st[c["seq"]]
        s["n"] += 1
        if bc is None:
            s["no_center"] += 1
        if br is None or iou2d(br, r) < 0.3:
            s["no_rect"] += 1
        if bc is not None and br is not None and list(bc) != list(br):
            s["differ"] += 1
            low[c["seq"]].append((c["vid"], fi, round(iou2d(bc, r), 2), round(iou2d(br, r), 2)))
print("seq   n  no_center  no_rect(<0.3)  differ")
for s in sorted(st):
    v = st[s]
    print(f"  {s} {v['n']:4d}  {v['no_center']:4d}  {v['no_rect']:4d}  {v['differ']:4d}   e.g. {low[s][:3]}")
tot = collections.Counter()
for v in st.values():
    tot.update(v)
print("total", dict(tot))
