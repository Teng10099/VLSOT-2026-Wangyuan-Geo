"""把两个预训练模型转成 ONNX（比赛要求 PyTorch 权重另附 ONNX）。

  python export_onnx.py yolo [--weights weights/yolo11x.pt]
      → weights/yolo11x_1088x1920.onnx。需要 ultralytics==8.3.40、torch、onnx。
        输入固定为 1x3x1088x1920：1920x1080 原图上下各补 4 像素，与 det.py 以 imgsz=1920 推理时的输入一致。
        输出 (1, 84, 42840)：cx, cy, w, h + 80 类 sigmoid 分数，后处理见 det_onnx.py。
  python export_onnx.py efn [--weights weights/EFN-model.best.h5]
      → weights/efn_color.onnx（输入 N x 224 x 224 x 3，float32，取值 0~1；输出 N x 15 的 softmax 概率）
        和 weights/efn_color_savedmodel/（TensorFlow SavedModel，saved_model.pb）。
        需要 tensorflow-cpu==2.11、tf2onnx==1.13、onnx==1.12（protobuf 3.19）。
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_common import WORK  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("which", choices=["yolo", "efn"])
parser.add_argument("--weights", default="")
parser.add_argument("--out", default="")
parser.add_argument("--opset", type=int, default=0, help="默认 yolo 17、efn 13")
args = parser.parse_args()

if args.which == "yolo":
    if os.path.isdir(os.path.join(WORK, "pydeps")):
        sys.path.insert(0, os.path.join(WORK, "pydeps"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # 不让 ultralytics 自动 pip 安装依赖
    from ultralytics import YOLO

    pt = args.weights or WORK + "/weights/yolo11x.pt"
    path = YOLO(pt).export(format="onnx", imgsz=[1088, 1920], opset=args.opset or 17, simplify=False,
                           dynamic=False, half=False, batch=1, device="cpu")
    out = args.out or WORK + "/weights/yolo11x_1088x1920.onnx"
    shutil.move(path, out)
    print("exported", out)
else:
    import tensorflow as tf
    import tf2onnx

    h5 = args.weights or WORK + "/weights/EFN-model.best.h5"
    model = tf.keras.models.load_model(h5)
    print("input", model.inputs[0].shape, "output", model.outputs[0].shape)
    sm_dir = WORK + "/weights/efn_color_savedmodel"
    model.save(sm_dir, include_optimizer=False)
    print("saved", sm_dir)
    out = args.out or WORK + "/weights/efn_color.onnx"
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=args.opset or 13, output_path=out)
    print("exported", out)
