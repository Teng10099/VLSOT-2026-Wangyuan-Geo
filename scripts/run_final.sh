#!/usr/bin/env bash
# 复现 v12f（2026-10-24 16:30 提交平台的版本）：检测 → 颜色 → 5 折交叉验证 → 测试集提交文件。
# 在仓库根目录运行：bash scripts/run_final.sh [all|det|color|cv|test]
# 运行前设置 VLSOT_DATA、VLSOT_CALIB（见 README“数据准备”）；权重放在 $VLSOT_WORK/weights/，缓存写到 $VLSOT_WORK/cache/。
# 已有缓存（权重包里的 cache/）时会跳过检测与颜色两步。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-python}
export VLSOT_WORK=${VLSOT_WORK:-$ROOT}
# 提交结果是在 2 线程 BLAS 下算出的；线程数不同时浮点求和顺序会变，个别数值可能有极小差异
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
CFG=$(cat "$ROOT/configs/v12f.args")
STEP=${1:-all}
cd "$ROOT/vlsot_geo"

if [ "$STEP" = all ] || [ "$STEP" = det ]; then
  # YOLO11x（COCO 预训练），1920 输入，FP16，需要 GPU 与 ultralytics==8.3.40；也可改用 det_onnx.py
  [ -f "$VLSOT_WORK/cache/dets_yolo11x_1920.json" ] || $PY det.py
fi
if [ "$STEP" = all ] || [ "$STEP" = color ]; then
  # EFN 颜色概率（CPU 即可）；--backend onnx 时只需要 onnxruntime
  [ -f "$VLSOT_WORK/cache/efn_color.pkl" ] || $PY efn_color.py --backend "${EFN_BACKEND:-keras}"
fi
if [ "$STEP" = all ] || [ "$STEP" = cv ]; then
  mkdir -p "$VLSOT_WORK/runs/cv"
  for k in 0 1 2 3 4; do
    $PY pipeline3.py --mode cv --fold $k $CFG --dump "$VLSOT_WORK/runs/cv/v12f_f$k.json" \
      > "$VLSOT_WORK/runs/cv/v12f_f$k.txt" 2>&1 &
  done
  wait
  $PY cv_agg.py v12f "$VLSOT_WORK"/runs/cv/v12f_f?.json
fi
if [ "$STEP" = all ] || [ "$STEP" = test ]; then
  # 用全部 227 段 TRAIN 拟合参数并写出 params_v12f.json，再对 10 段 TEST 生成 out/test_v12f/result.zip
  $PY pipeline3.py --mode test $CFG --tag v12f --dump_params "$VLSOT_WORK/out/params_v12f.json"
fi
