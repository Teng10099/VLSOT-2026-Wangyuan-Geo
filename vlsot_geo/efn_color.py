"""用 EFN 颜色分类器计算候选框的 15 类颜色概率，缓存成 pkl，供 pipeline3.py --color_model efn 使用。

EFN 是数据集作者为车辆颜色打初始标签时所用的分类器（来源与许可见 THIRD_PARTY_NOTICES.md）。
预处理与作者的标注脚本一致：PIL 裁框（坐标取整）→ 224x224 LANCZOS → /255，RGB，NHWC。
只用公开图像与我们自己的检测框，不读任何标签。
为控制 CPU 时间，只算置信度 ≥ --conf 且宽高都 ≥ 6 像素的框，每段只算第 0/14/29 帧和每 --every 帧一帧；
其余框的概率记为 NaN，pipeline3.py 会把它们当作“没有颜色信息”。

两种后端（输出一致性见 check_onnx.py）：
  --backend keras  读取 EFN-model.best.h5，需要 tensorflow-cpu==2.11
  --backend onnx   读取 export_onnx.py efn 转出的 efn_color.onnx，只需要 onnxruntime

缓存按“图像 → 框序号”存储，必须和 pipeline3.py 的 --dets 是同一份检测框。
用法：python efn_color.py --backend onnx --dets $VLSOT_WORK/cache/dets_yolo11x_onnx_test.json \
                          --splits test --out $VLSOT_WORK/cache/efn_color_onnx_test.pkl
"""
import argparse
import json
import os
import pickle

import numpy as np
from PIL import Image

from geo_common import DATA, WORK, load_clips

CLASSES = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red',
           'silver', 'tan', 'white', 'yellow']

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="keras", choices=["keras", "onnx"])
parser.add_argument("--model", default="", help="默认 weights/EFN-model.best.h5（keras）或 weights/efn_color.onnx（onnx）")
parser.add_argument("--dets", default=WORK + "/cache/dets_yolo11x_1920.json")
parser.add_argument("--out", default=WORK + "/cache/efn_color.pkl")
parser.add_argument("--splits", default="train,test")
parser.add_argument("--conf", type=float, default=float(os.environ.get("EFN_CONF", "0.25")))
parser.add_argument("--every", type=int, default=int(os.environ.get("EFN_EVERY", "3")))
parser.add_argument("--threads", type=int, default=int(os.environ.get("EFN_THREADS", "32")))
args = parser.parse_args()

if args.backend == "keras":
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    model = tf.keras.models.load_model(args.model or WORK + "/weights/EFN-model.best.h5")

    def predict(x):
        return model.predict(x, batch_size=64, verbose=0)
else:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model or WORK + "/weights/efn_color.onnx", opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def predict(x):
        return np.concatenate([sess.run(None, {inp: x[i:i + 64]})[0] for i in range(0, len(x), 64)])

dets = json.load(open(args.dets))
keys = set()
for split in args.splits.split(","):
    for c in load_clips(split):
        for fi in range(30):
            if fi in (0, 14, 29) or fi % args.every == 0:
                keys.add(f"{split}/{c['frames'][fi]['img']}")
keys = sorted(keys)
out = pickle.load(open(args.out, "rb"))["probs"] if os.path.exists(args.out) else {}
print("backend", args.backend, "images", len(keys), "already", len(out), flush=True)
batch, index = [], []


def flush():
    if not batch:
        return
    p = predict(np.stack(batch))
    for (k, j), row in zip(index, p):
        out[k][j] = row.astype(np.float32)
    batch.clear()
    index.clear()


def save():
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump({"names": CLASSES, "probs": out}, open(args.out + ".tmp", "wb"))
    os.replace(args.out + ".tmp", args.out)


for n, k in enumerate(keys):
    if k in out:
        continue
    boxes = dets[k]
    out[k] = np.full((len(boxes), len(CLASSES)), np.nan, dtype=np.float32)
    split, name = k.split("/")
    img = Image.open(f"{DATA}/{split}/img/{name}").convert("RGB")
    for j, b in enumerate(boxes):
        if b[4] < args.conf or b[2] - b[0] < 6 or b[3] - b[1] < 6:
            continue
        crop = img.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3]))).resize((224, 224), Image.LANCZOS)
        batch.append(np.asarray(crop, dtype=np.float32) / 255.0)
        index.append((k, j))
        if len(batch) >= 256:
            flush()
    if n % 200 == 0:
        flush()
        save()
        print(n, len(keys), flush=True)
flush()
save()
print("saved", args.out, len(out))
