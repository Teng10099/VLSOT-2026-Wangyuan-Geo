"""对 TRAIN/TEST 图像运行 COCO 预训练 YOLO11x（ultralytics 8.3.40，PyTorch 权重），缓存车辆检测框。

输出 json：{"train/001185.jpg": [[x1, y1, x2, y2, conf, cls], ...], ...}，cls 为 COCO 类别号
（2 car / 3 motorcycle / 5 bus / 7 truck），每张图的框按置信度从高到低排列。
只用公开输入图像，不读取任何标签。

提交结果（v12f）用的缓存 cache/dets_yolo11x_1920.json 就是本脚本的默认参数在 V100 上以 FP16 推理得到的。
不想安装 ultralytics（AGPL-3.0）时，可以改用 det_onnx.py 运行导出的 ONNX 模型，两者对比见 check_onnx.py。

用法：python det.py [--splits test,train] [--half 1] [--out cache/dets_yolo11x_1920.json]
"""
import argparse
import json
import os
import sys

from geo_common import DATA, WORK, load_clips

if os.path.isdir(os.path.join(WORK, "pydeps")):  # 可选：把 ultralytics 装在 $VLSOT_WORK/pydeps（pip install --target）
    sys.path.insert(0, os.path.join(WORK, "pydeps"))
from ultralytics import YOLO  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=WORK + "/weights/yolo11x.pt")
parser.add_argument("--imgsz", type=int, default=1920)
parser.add_argument("--conf", type=float, default=0.05)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--half", type=int, default=1, help="1=FP16 推理（与提交结果的缓存一致，需要 GPU）")
parser.add_argument("--device", default="0", help="GPU 编号，或 cpu")
parser.add_argument("--splits", default="test,train")
parser.add_argument("--out", default="", help="默认 cache/dets_<权重名>_<imgsz>.json")
args = parser.parse_args()

keys = []
for split in args.splits.split(","):
    for clip in load_clips(split):
        for f in clip["frames"]:
            keys.append(f"{split}/{f['img']}")
keys = sorted(set(keys))
print("images", len(keys), flush=True)

model = YOLO(args.weights)
name = os.path.splitext(os.path.basename(args.weights))[0]
out_path = args.out or f"{WORK}/cache/dets_{name}_{args.imgsz}.json"
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
result = {}
for start in range(0, len(keys), args.batch):
    chunk = keys[start:start + args.batch]
    paths = [f"{DATA}/{k.split('/')[0]}/img/{k.split('/')[1]}" for k in chunk]
    preds = model.predict(paths, imgsz=args.imgsz, conf=args.conf, classes=[2, 3, 5, 7],
                          verbose=False, device=args.device, half=bool(args.half))
    for k, p in zip(chunk, preds):
        b = p.boxes
        rows = [[round(float(v), 2) for v in xyxy] + [round(float(c), 4), int(k_)]
                for xyxy, c, k_ in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
        result[k] = rows
    if start % 800 == 0:
        print(start, flush=True)
json.dump(result, open(out_path, "w"))
print("saved", out_path)
