#!/usr/bin/env bash
# 技术报告消融表（PPT 第 4 页）的 9 行：每行在上一行基础上加一项改动，各跑 5 折交叉验证（227 段 TRAIN，按目标分组）。
# 在仓库根目录运行：bash scripts/run_ablation.sh      结果写到 $VLSOT_WORK/runs/ablation/，汇总打印到屏幕。
# 需要 cache/dets_yolo11x_1920.json、cache/efn_color.pkl，前两行还需要 cache/clip_color.pkl（clip_color.py）。
# 45 个进程并行，每个约占 1 GB 内存；机器较小时把 JOBS 调小。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-python}
export VLSOT_WORK=${VLSOT_WORK:-$ROOT}
# 报告中的数字是在 2 线程 BLAS 下跑出的；线程数不同时浮点求和顺序会变，个别片段可能有极小差异
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
JOBS=${JOBS:-45}
OUT="$VLSOT_WORK/runs/ablation"
mkdir -p "$OUT"
cd "$ROOT/vlsot_geo"

B="--stay_iou 0 --zmode median --ground global --color_every 3 --dir_w 2 --dir_min 0.1 --zray_w 0.75"
ATTR="--ordinal_mode gen --rel_feat 1 --range_tol 0.04 --range_tol_trunc 0.10"
EFN="--color_model efn --clip_floor 0.10 --ord_rmax 150"
declare -A CFG
NAMES=(a1_v9 a2_attr a3_v11c a4_gtrect a5_lin a6_bftrunc4 a7_bftrunc10 a8_bfall a9_v12f)
CFG[a1_v9]="$B --smooth state_ma --color_model clip"                                   # v9 基线：CLIP 颜色 + 射线定位
CFG[a2_attr]="$B --smooth state_ma --color_model clip $ATTR"                           # + 复刻生成规则的名次/最近车方位、距离死区
CFG[a3_v11c]="$B --smooth state_ma $ATTR $EFN"                                         # + EFN 颜色、名次框距上限、概率下限（v11c）
CFG[a4_gtrect]="${CFG[a3_v11c]} --gt_match rect"                                       # + 训练 GT↔检测框按投影矩形匹配
CFG[a5_lin]="$B --smooth state_lin $ATTR $EFN --gt_match rect"                         # + 运动目标方位角局部线性平滑
CFG[a6_bftrunc4]="${CFG[a5_lin]} --center_mode boxfit_trunc"                           # + boxfit（仅截断框，边距 4 px）
CFG[a7_bftrunc10]="${CFG[a5_lin]} --center_mode boxfit_trunc --bf_margin 10"           # + boxfit（仅截断框，边距 10 px）
CFG[a8_bfall]="${CFG[a5_lin]} --center_mode boxfit --bf_margin 10"                     # + boxfit（全部帧）
CFG[a9_v12f]="${CFG[a8_bfall]} --range_interp quad --bf_zsig 0.6"                      # + 二次距离插值、高度先验放宽（v12f）

for name in "${NAMES[@]}"; do
  for k in 0 1 2 3 4; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    $PY pipeline3.py --mode cv --fold $k ${CFG[$name]} --dump "$OUT/${name}_f$k.json" > "$OUT/${name}_f$k.txt" 2>&1 &
  done
done
wait
for name in "${NAMES[@]}"; do
  $PY cv_agg.py "$name" "$OUT/${name}"_f?.json
done
# 报告里引用的配对比较（bootstrap 95% 区间）与截断片段分析
$PY cv_pair.py "$OUT/a3_v11c_f?.json" "$OUT/a9_v12f_f?.json"
$PY cv_pair.py "$OUT/a8_bfall_f?.json" "$OUT/a9_v12f_f?.json"
PYTHONPATH=. $PY analysis/ana_trunc_cv.py "$OUT/a5_lin_f?.json" "$OUT/a8_bfall_f?.json" | tail -4
