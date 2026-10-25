"""测试片段可视化（只用图像、文本与我们的预测）：红框=选中检测框，绿色线框=预测 3D 框（文本尺寸 + 文本航向）投影。
可对比两个版本：python viz4.py out/test_A [out/test_B]   B 的 3D 框画成青色。输出 <A>/v4_<vid>.jpg"""
import json
import sys

import cv2
import numpy as np

from boxfit import corners
from geo_common import image_path, load_calib, load_clips, parse_text, project

EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
runs = sys.argv[1:]
res = [{c: np.loadtxt(f"{r}/result/{c}.txt") for c in [x["vid"] for x in load_clips("test")]} for r in runs]
tracks = json.load(open(f"{runs[0]}/tracks.json"))
colors = [(0, 255, 0), (255, 255, 0)]
for c in load_clips("test"):
    t = parse_text(c["desc"])
    tiles = []
    for fi in [0, 10, 20, 29]:
        img = cv2.imread(image_path(c, fi))
        calib = load_calib(c["frames"][fi]["img"])
        b = tracks[c["vid"]][fi]
        if b is not None:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        for k, rr in enumerate(res):
            p = rr[c["vid"]]
            mv = p[min(29, fi + 3), 4:6] - p[max(0, fi - 3), 4:6]
            yaw = np.arctan2(mv[1], mv[0]) if np.hypot(*mv) > 0.5 else np.radians(t["azimuth"] or 0.0)
            P = corners(p[fi, 4:7][None], p[fi, 1:4], yaw)[0]
            uv, z = project(P, calib)
            if (z > 0.5).all():
                for i, j in EDGES:
                    cv2.line(img, tuple(int(v) for v in uv[i]), tuple(int(v) for v in uv[j]), colors[k], 3)
        cv2.putText(img, f"{c['vid']} f{fi}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
        # 以目标为中心裁一块放大（便于看对齐），再拼上全图缩略
        if b is not None:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            half = max(b[2] - b[0], b[3] - b[1]) * 1.2 + 60
            x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
            x2, y2 = int(min(1920, cx + half)), int(min(1080, cy + half))
            crop = cv2.resize(img[y1:y2, x1:x2], (450, 450))
        else:
            crop = np.zeros((450, 450, 3), np.uint8)
        tiles.append(np.concatenate([cv2.resize(img, (800, 450)), crop], 1))
    grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    cv2.imwrite(f"{runs[0]}/v4_{c['vid']}.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 75])
print("done")
