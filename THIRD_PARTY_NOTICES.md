# 第三方组件与数据说明（Third-party notices）

本仓库的源代码按 Apache License 2.0 发布（见 `LICENSE`）。下列模型、库和数据由第三方提供，**不随本仓库分发**，也不受本仓库许可证约束；使用时请遵守各自的许可。

| 组件 | 用途 | 来源 | 许可 | 在本仓库中的使用方式 |
|---|---|---|---|---|
| YOLO11x 预训练权重 `yolo11x.pt` | 车辆检测 | Ultralytics，<https://huggingface.co/Ultralytics/YOLO11>（SHA256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`） | AGPL-3.0 | 不包含在仓库中；权重包里另附，原样分发并附许可说明 |
| `ultralytics==8.3.40`（Python 包） | `det.py` 的 PyTorch 推理、`export_onnx.py yolo` 的导出 | <https://github.com/ultralytics/ultralytics> | AGPL-3.0 | 只在这两个可选脚本里 `import`，不随仓库分发；`det_onnx.py` 不依赖它 |
| `yolo11x_1088x1920.onnx` | `det_onnx.py` 推理 | 由上面的 `yolo11x.pt` 用 `export_onnx.py yolo` 转换 | 随原权重，AGPL-3.0 | 不包含在仓库中；权重包里另附 |
| EFN 车辆颜色分类器 `EFN-model.best.h5` | 候选框颜色概率（`efn_color.py`） | 数据集作者为车辆颜色打初始标签时所用的分类器 | 未公开声明 | 不包含在仓库中；需经权利人同意后才能另行分享，见 README“权重下载” |
| CLIP ViT-B/16 | 只用于 v9 对照实验的颜色概率（`clip_color.py`） | OpenAI，<https://huggingface.co/openai/clip-vit-base-patch16> | MIT | 不包含在仓库中，可自行下载 |
| VLSOT 数据集（官方训练集与测试集） | 输入与 TRAIN 标签 | VLP 平台 <http://vlp.chd.edu.cn/> | CC BY-NC-SA 3.0，仅限比赛非商业使用，禁止二次分发 | 不包含在仓库与权重包中 |
| V2X-Seq-SPD 路侧相机标定 | 相机内参与外参（官方数据包不含标定） | V2X-Seq（DAIR-V2X），<https://github.com/AIR-THU/DAIR-V2X-Seq> | 以数据集官方条款为准 | 不包含在仓库中；代码只读取 `camera_intrinsic/` 与 `virtuallidar_to_camera/`，不读取其任何标注 |

说明：

1. 本仓库的代码（`vlsot_geo/`、`scripts/`、`docs/` 等）由望远队编写。名次、九宫格区域、与最近车的相对方位等属性的计算方式按数据集描述生成规则重新实现（例如名次按“格子左上角到框底边中点的像素距离”排序、相对方位以 3.7 m 为阈值），没有复制数据集作者的代码。
2. `det.py` 和 `export_onnx.py yolo` 在运行时导入 AGPL-3.0 许可的 `ultralytics`。只想使用宽松许可依赖的读者可以用 `det_onnx.py`（只依赖 onnxruntime / OpenCV / NumPy），但 YOLO11x 权重及其 ONNX 转换文件本身仍按 AGPL-3.0 发布。
3. 测试阶段不读取任何测试标签；所有参数只在官方 TRAIN 标签上拟合。
