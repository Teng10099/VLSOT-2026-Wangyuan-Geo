"""用导出的 YOLO11x ONNX 模型（export_onnx.py yolo）做车辆检测，只依赖 onnxruntime / OpenCV / NumPy，不需要 ultralytics。

预处理与后处理按 ultralytics 8.3.40 predict 的实现复写：
  letterbox：1920x1080 原图不缩放，上下各补 4 像素灰边（114）到 1088x1920；BGR→RGB，CHW，/255，float32。
  输出 (1, 84, N)：cx, cy, w, h + 80 类分数（模型内已做 sigmoid）。
  每个框取最高分类别；分数 > conf；只保留 car/motorcycle/bus/truck（COCO 2/3/5/7）；
  按类别偏移（类别号 × 7680）做 NMS（IoU > 0.7 抑制）；最多 300 个；减去灰边并裁到图像范围内。
输出格式与 det.py 相同：{"test/000019.jpg": [[x1, y1, x2, y2, conf, cls], ...]}，按置信度从高到低。
只用公开输入图像，不读取任何标签。

用法：python det_onnx.py --splits test --out $VLSOT_WORK/cache/dets_yolo11x_onnx_test.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from geo_common import DATA, WORK, load_clips

CLASSES = (2, 3, 5, 7)
MAX_WH = 7680  # ultralytics 的类别偏移量

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=WORK + "/weights/yolo11x_1088x1920.onnx")
parser.add_argument("--splits", default="test")
parser.add_argument("--out", default=WORK + "/cache/dets_yolo11x_onnx.json")
parser.add_argument("--conf", type=float, default=0.05)
parser.add_argument("--iou", type=float, default=0.7)
parser.add_argument("--max_det", type=int, default=300)
parser.add_argument("--threads", type=int, default=0, help="onnxruntime 线程数，0=默认")
parser.add_argument("--cuda", type=int, default=0, help="1=优先用 CUDAExecutionProvider（需要 onnxruntime-gpu）")
parser.add_argument("--limit", type=int, default=0, help=">0 时只处理前若干张图（快速自检）")
args = parser.parse_args()


def letterbox(img, new_hw):
    """返回补边后的图像、缩放比例 r、(左, 上) 补边像素；与 ultralytics LetterBox(center=True) 相同的取整方式。"""
    h, w = img.shape[:2]
    r = min(new_hw[0] / h, new_hw[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (new_hw[1] - new_unpad[0]) / 2, (new_hw[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def nms(boxes, scores, thr):
    """贪心 NMS（与 torchvision.ops.nms 相同：IoU 严格大于阈值才抑制，面积不加 1）。返回保留下标，按分数降序。"""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (area[i] + area[rest] - inter + 1e-12)
        order = rest[iou <= thr]
    return np.array(keep, dtype=np.int64)


def postprocess(pred, r, pad, img_hw):
    """pred: (84, N) → [[x1, y1, x2, y2, conf, cls], ...]"""
    p = pred.T
    scores = p[:, 4:]
    conf = scores.max(1)
    cls = scores.argmax(1)
    m = conf > args.conf
    p, conf, cls = p[m], conf[m], cls[m]
    m = np.isin(cls, CLASSES)
    p, conf, cls = p[m], conf[m], cls[m]
    if not len(p):
        return []
    xy, wh = p[:, :2], p[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    if len(boxes) > 30000:
        top = np.argsort(-conf, kind="stable")[:30000]
        boxes, conf, cls = boxes[top], conf[top], cls[top]
    keep = nms(boxes + cls[:, None] * MAX_WH, conf, args.iou)[:args.max_det]
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_hw[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_hw[0])
    return [[round(float(v), 2) for v in b] + [round(float(c), 4), int(k)] for b, c, k in zip(boxes, conf, cls)]


def main():
    opts = ort.SessionOptions()
    if args.threads > 0:
        opts.intra_op_num_threads = args.threads
    providers = (["CUDAExecutionProvider"] if args.cuda else []) + ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, opts, providers=providers)
    inp = sess.get_inputs()[0]
    new_hw = tuple(int(v) for v in inp.shape[2:4])
    print("model", args.model, "input", inp.name, inp.shape, "providers", sess.get_providers(), flush=True)
    keys = sorted({f"{split}/{f['img']}" for split in args.splits.split(",") for c in load_clips(split)
                   for f in c["frames"]})
    if args.limit > 0:
        keys = keys[:args.limit]
    print("images", len(keys), flush=True)
    result, t0 = {}, time.time()
    for n, k in enumerate(keys):
        split, name = k.split("/")
        img = cv2.imread(f"{DATA}/{split}/img/{name}")
        x, r, pad = letterbox(img, new_hw)
        x = np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        pred = sess.run(None, {inp.name: x})[0][0]
        result[k] = postprocess(pred, r, pad, img.shape[:2])
        if n % 50 == 0:
            print(n, len(keys), "%.1fs" % (time.time() - t0), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"))
    print("saved", args.out, "images", len(result), "boxes", sum(len(v) for v in result.values()))


if __name__ == "__main__":
    main()
