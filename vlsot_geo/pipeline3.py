"""VLSOT 文本约束几何流水线（Tenn-VLSOT-Geo）主程序：目标选择 + 三维定位 + 时序平滑。

流程（参数说明见各 --help；最终配置见 configs/v12f.args）：
  1. 文本解析：尺寸 l/w/h、第 0/14/29 帧距离、航向与朝向词、九宫格区域、名次、截断、最近车关系。
  2. 候选：每帧的检测框（det.py / det_onnx.py 预先缓存），颜色概率（efn_color.py / clip_color.py 预先缓存）。
  3. 候选打分：线性模型，softmax 排序损失，只用 TRAIN 中与 GT 对应的检测框训练（train_scorer）。
  4. 全局跟踪：30 帧 Viterbi（track_clip），二元代价 = 2D IoU + 运动方向与文本航向的一致性。
  5. 三维定位：--center_mode boxfit，已知文本尺寸/航向/距离，求 3D 框投影外接矩形与检测框未截断边对齐的中心（boxfit.py）。
  6. 时序平滑：停车目标取 30 帧中位数；运动目标保持文本距离曲线，只对方位角做局部线性平滑（--smooth state_lin）。

运行模式：
  --mode cv    在 227 段 TRAIN 上按目标 (sequence, track) 分组做 5 折交叉验证，评估第 --fold 折（--dump 写逐段结果）。
  --mode test  用全部 TRAIN 拟合参数，对 TEST 输出 result/<视频ID>.txt 与 result.zip；
               --dump_params 把拟合出的全部参数写成 json；--load_params 直接读取该 json，不再需要 TRAIN 标签。

开发过程中逐步加入的选项（默认值即旧行为，最终配置把它们打开）：
  v9   --ground global 全局地平面；--color_model clip 颜色；--dir_w 运动方向一致性；--zray_w 0.75。
  v10  按数据集描述生成脚本复刻属性定义：--ordinal_mode gen 名次 = 同一九宫格内按“格子左上角 → 框底边中点”的像素距离排序；
       --rel_feat 与最近车的相对方位（3.7 m 阈值）；--range_tol / --range_tol_trunc 距离死区。
  v11c --color_model efn 数据集作者标注颜色所用的 EFN 分类器；--ord_rmax 150 名次只计 150 m 内的框；--clip_floor 颜色概率下限。
  v12f --gt_match rect 训练时 GT 与检测框按投影矩形 IoU 匹配；--smooth state_lin；--center_mode boxfit --bf_margin 10；
       --range_interp quad 三处文本距离过三点二次插值；--bf_zsig 0.6 高度先验标准差。
  另有若干试过但未采用的选项（--seq_bias、--yaw_motion、--blend_ray 等），默认关闭，结果见 README。

所有拟合只用 TRAIN 标签；TEST 只用图像、文本描述、json 中的 color/state/type 字段、相机标定和检测框。
"""
import argparse
import collections
import json
import math
import os
import re
import zipfile

import cv2
import numpy as np

from boxfit import fit_center as bf_fit_center
from boxfit import rects as bf_rects
from boxfit import trunc_flags
from geo_common import (WORK, evaluate, evaluate_platform, gt_array, image_path, load_calib, load_clips,
                        parse_text, pixel_rays, project, read_ids)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="seen", choices=["val", "seen", "test", "cv"],
                    help="cv=按目标 (seq,track) 分 5 折，评估第 --fold 折（全部 227 段轮流作验证）")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--nfold", type=int, default=5)
parser.add_argument("--range_tol", type=float, default=0.0,
                    help="距离特征的死区：|r_plane-文本距离|/文本距离 小于该值不计代价（地平面误差约 3%%）")
parser.add_argument("--range_tol_trunc", type=float, default=-1.0, help="截断框的距离死区（<0 则同 range_tol）")
parser.add_argument("--dets", default=WORK + "/cache/dets_yolo11x_1920.json", help="检测框缓存（det.py / det_onnx.py 输出）")
parser.add_argument("--efn_cache", default=WORK + "/cache/efn_color.pkl",
                    help="EFN 颜色概率缓存（efn_color.py 输出，必须与 --dets 是同一份检测框）")
parser.add_argument("--clip_cache", default=WORK + "/cache/clip_color.pkl", help="CLIP 颜色概率缓存（clip_color.py 输出）")
parser.add_argument("--ground", default="global_offset",
                    choices=["seq_plane", "global", "global_offset", "track_offset"],
                    help="track_offset=每个目标先取底面高度中位数，再按 1/方差 加权得到相机偏移，防止单个噪声目标主导")
parser.add_argument("--zsrc", default="plane", choices=["plane", "ray", "avg"])
parser.add_argument("--zray_w", type=float, default=-1.0,
                    help=">=0 时 z = w*射线高度 + (1-w)*地平面高度，覆盖 --zsrc（avg 相当于 0.5）")
parser.add_argument("--iou_w", type=float, default=2.0)
parser.add_argument("--anchor_w", type=float, default=1.0)
parser.add_argument("--other_w", type=float, default=0.3)
parser.add_argument("--anchor_mid", type=int, default=14)
parser.add_argument("--smooth", default="state_ma", choices=["none", "state", "state_ma", "state_lin"],
                    help="state=停车取中位数+运动二次拟合；state_ma=停车取中位数+运动5帧滑动平均；"
                         "state_lin=停车取中位数+运动目标距离不动、方位角局部线性平滑（无首尾偏差）")
parser.add_argument("--smooth_k", type=int, default=3, help="state_lin 的半窗口（帧）")
parser.add_argument("--range_interp", default="lin", choices=["lin", "quad"],
                    help="文本三个距离（第 0/14/29 帧）插值成逐帧距离：lin=分段线性；quad=过三点二次曲线")
parser.add_argument("--zmode", default="median", choices=["frame", "median"], help="median=整段取高度中位数")
parser.add_argument("--switch_cost", type=float, default=6.0, help="相邻帧框不重叠（换车）时的固定代价")
parser.add_argument("--stay_iou", type=float, default=0.3, help="相邻帧 IoU 低于该值视为换车；设 0 恢复 v3 行为")
parser.add_argument("--color_every", type=int, default=3, help="每隔几帧计算一次颜色特征；0=只在关键帧")
parser.add_argument("--seen_split", default="clip", choices=["clip", "track"])
parser.add_argument("--loo_track", type=int, default=0,
                    help="1=seen 模式下评估每个片段时，地面偏移不使用同一目标(seq,track)的标签，模拟测试“新目标”条件")
parser.add_argument("--color_model", default="knn", choices=["knn", "clip", "efn", "efn_clip"],
                    help="efn=数据集作者的颜色分类器（缓存 cache/efn_color.pkl）；efn_clip=两者 -log 概率取平均")
parser.add_argument("--dir_w", type=float, default=0.0,
                    help="运动目标的方向一致性代价权重：相邻帧 3D 位移与文本 azimuth（virtuallidar 下的航向角）"
                         "夹角越大代价越高；0=关闭")
parser.add_argument("--dir_min", type=float, default=0.3, help="相邻帧位移小于该值(米)时不计方向代价")
parser.add_argument("--ordinal_mode", default="old", choices=["old", "gen"],
                    help="gen=按数据生成脚本的定义算名次（格子左上角到框底边中点的距离）；old=旧的按距离/纵坐标排序")
parser.add_argument("--ord_conf", type=float, default=0.3, help="参与名次计算的检测框最低置信度")
parser.add_argument("--ord_rmax", type=float, default=1e9,
                    help="参与名次/最近车计算的检测框的最大平面距离（米）；远处未标注的小车会把名次抬高，TRAIN 上 150 最好")
parser.add_argument("--rel_feat", type=int, default=0, help="1=加入“与最近车辆的方位关系”特征（生成脚本 space_relate3d）")
parser.add_argument("--near_color_feat", type=int, default=0, help="1=加入“最近车辆颜色”特征（需要 rel_feat 的最近车计算）")
parser.add_argument("--trunc_z", type=int, default=0, help="1=框贴上/下图像边界时 z 改用地平面（截断框中心偏高）")
parser.add_argument("--heading_mode", default="all", choices=["all", "first"],
                    help="all=三段朝向（首段数值 azimuth + 中/末段朝向词）插值；first=全程只用首段 azimuth")
parser.add_argument("--clip_floor", type=float, default=0.0,
                    help="CLIP 颜色概率下限：低于该值一律视为“不是该颜色”，避免 0.05 与 0.18 这类无意义差异主导选择")
parser.add_argument("--oracle", type=int, default=0, help="1=用 GT 框代替选择（只用于在 TRAIN 上诊断几何上限）")
parser.add_argument("--center_mode", default="ray",
                    choices=["ray", "boxfit", "boxfit_trunc", "boxfit_near", "boxfit_far"],
                    help="中心估计：ray=框中心(+全局偏移)射线与文本距离圆柱求交（v11 及以前）；"
                         "boxfit=已知文本尺寸/航向/距离，求 3D 框投影外接矩形与检测框未截断边对齐的中心；"
                         "boxfit_trunc=只对贴图像边界的框用 boxfit；boxfit_near=截断或距离 < --bf_near 时用 boxfit；"
                         "boxfit_far=截断或距离 >= --bf_near 时用 boxfit")
parser.add_argument("--bf_near", type=float, default=30.0)
parser.add_argument("--bf_margin", type=float, default=4.0, help="框边距图像边界小于该像素数视为该边被截断")
parser.add_argument("--bf_zsig", type=float, default=0.3, help="boxfit 中 z 的地平面先验标准差（米）")
parser.add_argument("--bf_bias", type=int, default=0, help="1=各边使用 TRAIN 上学到的偏置（投影框边 - 检测框边）")
parser.add_argument("--bf_hc", type=float, default=0.0, help="boxfit z 先验 = 地平面 + h/2 + bf_hc")
parser.add_argument("--seq_bias", type=float, default=0.0,
                    help=">0 时减去同一 sequence（或同一相机）TRAIN 片段上学到的中心系统偏差（横向、高度）；"
                         "数值为收缩伪计数 K，权重 = n/(n+K)，n 为该组的 TRAIN 片段数；0=关闭")
parser.add_argument("--bias_level", default="seqcam", choices=["seq", "cam", "seqcam"],
                    help="seqcam=优先用同 sequence 的偏差，没有则用同一相机（外参平移相同）的偏差")
parser.add_argument("--yaw_motion", type=int, default=0, help="1=运动目标用第一遍轨迹方向作为 boxfit 航向再算一遍")
parser.add_argument("--blend_ray", type=float, default=0.0,
                    help="未截断帧的中心 = (1-a)*boxfit + a*ray；0=只用 boxfit（v12f）")
parser.add_argument("--yaw_motion_min", type=float, default=0.5, help="±3 帧位移小于该值（米）时保留文本航向")
parser.add_argument("--gt_match", default="center", choices=["center", "rect"],
                    help="训练/oracle 时 GT 对应检测框的匹配：center=含 GT 中心投影的最小框；"
                         "rect=中心匹配失败时（截断目标）再用 GT 3D 框投影矩形的最大 IoU 框")
parser.add_argument("--fit_frac", type=float, default=1.0, help="<1 时随机只用这部分拟合片段（看训练量是否是瓶颈）")
parser.add_argument("--fit_seed", type=int, default=0)
parser.add_argument("--dump", default="", help="验证模式下把每段的逐帧结果写成 json，供失败分析")
parser.add_argument("--dump_params", default="", help="把在 TRAIN 上拟合的全部参数写成 json（权重包里的 params_v12f.json）")
parser.add_argument("--load_params", default="",
                    help="只用于 --mode test：读取 --dump_params 写出的参数，跳过拟合，不读取任何 TRAIN 标签")
parser.add_argument("--out", default=WORK + "/out")
parser.add_argument("--tag", default="")
args = parser.parse_args()

DETS = json.load(open(args.dets))
CLIP = None
EFN = None
if args.color_model in ("clip", "efn_clip"):
    import pickle
    CLIP = pickle.load(open(args.clip_cache, "rb"))
    CLIP_INDEX = {n: i for i, n in enumerate(CLIP["names"])}
if args.color_model in ("efn", "efn_clip"):
    import pickle
    EFN = pickle.load(open(args.efn_cache, "rb"))
    EFN_INDEX = {n: i for i, n in enumerate(EFN["names"])}
W_IMG, H_IMG = 1920, 1080
TYPE_TO_COCO = {"Car": {2}, "Van": {2, 7}, "Truck": {7, 5}, "Bus": {5, 7}}
COLORS = ["white", "black", "silver grey", "red", "blue", "yellow-orange", "green", "brown", "other"]
ANCHORS = {0: 0, args.anchor_mid: 1, 29: 2}
N_FEAT = 11
FEAT_NAMES = ["range", "cls", "conf", "region", "trunc", "color", "aspect", "anchor", "ordinal", "relation",
              "near_color"]


# ---------------------------------------------------------------- 文本解析
REGION_RE = re.compile(r"((?:upper|top|lower|bottom|middle|central|left|right|center)"
                       r"(?:[\s-]+(?:upper|top|lower|bottom|middle|central|left|right|center))*)\s+"
                       r"(?:corner|section|area|part|region|portion|side)")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
            "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20}
_ORD_WORDS = "|".join(sorted(ORDINALS, key=len, reverse=True))


def region_of(phrase):
    row = col = None
    for w in phrase.split():
        if w in ("upper", "top"):
            row = 0
        elif w in ("lower", "bottom"):
            row = 2
        elif w == "left":
            col = 0
        elif w == "right":
            col = 2
    return (1 if row is None else row, 1 if col is None else col)


def parse_ordinal(seg):
    m = re.search(rf"({_ORD_WORDS})\s+(?:\w+\s+)?(?:vehicle|car|truck|van|bus|object|order|position|one|item|entity)", seg)
    if not m:
        m = re.search(rf"(?:order|position)\s+(?:is\s+|of\s+)?(?:the\s+)?({_ORD_WORDS})", seg)
    if m:
        return (ORDINALS[m.group(1)], "top" if "from the top" in seg else "near")
    m = re.search(r"(?:order|position|number)\s+(?:of\s+)?(\d{1,2})\b", seg) or \
        re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+(?:vehicle|car|object|order|position)", seg)
    if m:
        return (int(m.group(1)), "near")
    return None


def gen_cell(b):
    """与数据生成脚本 01Re/02Re/03Re4color.py 一致：九宫格 = ceil(中心/格宽) - 1。"""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return (min(2, max(0, math.ceil(cy / (H_IMG / 3)) - 1)), min(2, max(0, math.ceil(cx / (W_IMG / 3)) - 1)))


def gen_ordinal_ranks(boxes):
    """生成脚本 order_judege 的定义：同一格内，按“格子左上角 → 框底边中点”的像素距离升序排名（1 起）。
    boxes 为参与排名的框（车辆，已做类别无关 NMS）。返回每个框的名次。"""
    cells = [gen_cell(b) for b in boxes]
    ranks = []
    for i, b in enumerate(boxes):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        corner = (W_IMG / 3 * math.floor(cx / (W_IMG / 3)), H_IMG / 3 * math.floor(cy / (H_IMG / 3)))
        same = [j for j in range(len(boxes)) if cells[j] == cells[i]]
        dist = {j: math.hypot((boxes[j][0] + boxes[j][2]) / 2 - corner[0], boxes[j][3] - corner[1]) for j in same}
        ranks.append(1 + sum(1 for j in same if dist[j] < dist[i]))
    return ranks


def nms_agnostic(boxes, thr=0.7):
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][4])
    keep = []
    for i in order:
        if all(iou2d(boxes[i], boxes[k]) < thr for k in keep):
            keep.append(i)
    return keep


def parse_chunks(desc):
    text = desc.lower()
    cuts = [m.start() for m in re.finditer(r"as the video (?:progresses|continues)|towards the end|by the end|"
                                           r"at the end|in the end|finally", text)]
    bounds = [0] + cuts[:2] + [len(text)]
    while len(bounds) < 4:
        bounds.insert(-1, len(text))
    chunks = []
    for k in range(3):
        seg = text[bounds[k]:bounds[k + 1]] if bounds[k] < bounds[k + 1] else ""
        reg = None
        for m in REGION_RE.finditer(seg):
            pre = seg[max(0, m.start() - 25):m.start()]
            if "relative" in pre or "positioned to the" in pre:
                continue
            reg = region_of(m.group(1))
            break
        if reg is None and ("middle of the video" in seg or "center of the video" in seg):
            reg = (1, 1)
        trunc = None
        if re.search(r"not truncated|untruncated|non-truncated|non truncated|no truncation|without (?:any )?truncation|"
                     r"no longer truncated|not being truncated|isn't truncated|is not truncated|free of truncation", seg):
            trunc = 0
        elif "truncat" in seg:
            trunc = 1
        chunks.append({"region": reg, "trunc": trunc, "ordinal": parse_ordinal(seg), "orient": parse_orient(seg),
                       "rel": parse_relation(seg), "near_color": parse_near_color(seg)})
    return chunks


_NEAR_COLOR_WORDS = [("silver grey", r"silver|grey|gray"), ("white", r"white"), ("black", r"black"),
                     ("red", r"\bred\b|pink"), ("blue", r"blue|purple"), ("yellow-orange", r"yellow|orange|gold"),
                     ("green", r"green"), ("brown", r"brown|beige|\btan\b")]


def parse_near_color(seg):
    """最近车辆的颜色（生成脚本 spatial_relation_nearest_vehicle_color）。"""
    m = re.search(r"nearest|closest|adjacent|accompanied|next to|followed|alongside", seg)
    if not m:
        return None
    win = seg[max(0, m.start() - 10):m.start() + 70]
    best = None
    for name, pat in _NEAR_COLOR_WORDS:
        mm = re.search(pat, win)
        if mm and (best is None or mm.start() < best[1]):
            best = (name, mm.start())
    return best[0] if best else None


# 与最近车辆的空间关系（生成脚本 space_relate3d：描述“目标相对最近车辆”的方位，x 向前、y 向左，阈值 3.7 m）。
# 用 (dx 符号, dy 符号) 表示：left=(0,+1) right=(0,-1) top=(+1,0) under=(-1,0) lower right=(-1,-1) ...
REL_VEC = {"left": (0, 1), "right": (0, -1), "top": (1, 0), "under": (-1, 0),
           "lower right": (-1, -1), "lower left": (-1, 1), "upper right": (1, -1), "upper left": (1, 1)}
_REL_PATTERNS = [("lower right", r"lower[\s-]+right"), ("lower left", r"lower[\s-]+left"),
                 ("upper right", r"upper[\s-]+right"), ("upper left", r"upper[\s-]+left"),
                 ("under", r"underside|beneath|\bbelow\b|\bbottom\b|lower side"),
                 ("top", r"\btop\b|\babove\b|upper side"), ("right", r"\bright\b"), ("left", r"\bleft\b")]
_MOTION_PHRASE = re.compile(r"(?:moving|traveling|travelling|driving|heading|facing|oriented|going|turning)\s+"
                            r"(?:in\s+the\s+|to\s+the\s+|towards?\s+the\s+)?(?:left|right|front|back|forward|backward|"
                            r"opposite|same)(?:\s+direction)?")


def parse_relation(seg):
    m = re.search(r"nearest|closest|adjacent|accompanied|next to|followed|alongside|neighbo", seg)
    if not m:
        return None
    tail = _MOTION_PHRASE.sub(" ", seg[m.start():m.start() + 260])
    for name, pat in _REL_PATTERNS:
        if re.search(pat, tail):
            return name
    return None


def space_relate3d(self_xy, near_xy, thr=3.7):
    sx, sy = self_xy
    nx, ny = near_xy
    if abs(sx - nx) <= thr:
        return "left" if sy > ny else "right"
    if abs(sy - ny) <= thr:
        return "top" if sx > nx else "under"
    if nx > sx:
        return "lower right" if sy < ny else "lower left"
    return "upper right" if sy < ny else "upper left"


# 朝向词 -> virtuallidar 航向角（度）。由 TRAIN 标签核对：azimuth 与运动方向中位差 1.8°，
# front/forward=0（远离相机），back/backward=180，left=+90，right=-90。
ORIENT_DEG = {"front": 0.0, "forward": 0.0, "forwards": 0.0, "back": 180.0, "backward": 180.0, "backwards": 180.0,
              "left": 90.0, "leftward": 90.0, "right": -90.0, "rightward": -90.0}
ORIENT_RE = re.compile(r"(?:fac(?:es|ing)|oriented|orientation|pointing)\s+(?:to(?:wards)?\s+)?(?:the\s+)?"
                       r"(front|forwards?|back|backwards?|left|right)\b|\b(leftward|rightward|forward|backward)\s+orientation")


def parse_orient(seg):
    m = ORIENT_RE.search(seg)
    if not m:
        return None
    return ORIENT_DEG[m.group(1) or m.group(2)]


def color_of(clip):
    c = (clip.get("color") or "").lower().replace("-", " ").strip()
    for name in COLORS:
        n = name.replace("-", " ")
        if n in c or (c and c in n):
            return name
    return "other"


# EFN 颜色分类器（数据集作者给颜色打初始标签所用）15 类 -> 文本颜色
EFN_GROUPS = {"white": ["white"], "black": ["black"], "silver grey": ["silver", "grey"], "red": ["red", "pink"],
              "blue": ["blue", "purple"], "yellow-orange": ["yellow", "orange", "gold"], "green": ["green"],
              "brown": ["brown", "tan", "beige"]}


# ---------------------------------------------------------------- 颜色
def color_feature(img, box):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    crop = img[max(0, y1 + h // 4):min(H_IMG, y2 - h // 4), max(0, x1 + w // 4):min(W_IMG, x2 - w // 4)]
    if crop.size == 0:
        return np.zeros(32, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hh, ss, vv = hsv[:, 0] / 180.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    n = max(len(hh), 1)
    f = np.concatenate([np.histogram(hh, bins=12, range=(0, 1), weights=ss * vv)[0] / n * 4,
                        np.histogram(ss, bins=6, range=(0, 1))[0] / n,
                        np.histogram(vv, bins=8, range=(0, 1))[0] / n,
                        crop.reshape(-1, 3).mean(0) / 255.0, [ss.mean(), vv.mean(), vv.std()]])
    return f.astype(np.float32)


class ColorKNN:
    def __init__(self, feats, labels, k=25, keys=None):
        self.X, self.y, self.k = np.array(feats), np.array(labels), k
        self.keys = list(keys) if keys is not None else [None] * len(self.y)

    def without(self, key):
        """去掉某个目标 (seq, track) 的样本，用于 seen 留一目标评估（测试目标都是新目标）。"""
        keep = [i for i, k in enumerate(self.keys) if k != key]
        return ColorKNN(self.X[keep], self.y[keep], self.k, [self.keys[i] for i in keep])

    def prob(self, f, name):
        idx = np.argsort(np.linalg.norm(self.X - f, axis=1))[:self.k]
        return (np.sum(self.y[idx] == name) + 0.5) / (self.k + 0.5 * len(COLORS))


# ---------------------------------------------------------------- 几何
def contains(b, uv):
    return b[0] <= uv[0] <= b[2] and b[1] <= uv[1] <= b[3]


def ray_center(box, calib, offset):
    w, h = box[2] - box[0], box[3] - box[1]
    o, d = pixel_rays([[(box[0] + box[2]) / 2 + offset[0] * w, (box[1] + box[3]) / 2 + offset[1] * h]], calib)
    return o, d[0]


def on_plane(o, d, plane, hh):
    a, b, c = plane
    n = np.array([a, b, -1.0])
    den = d @ n
    if abs(den) < 1e-9:
        return None
    s = -(o @ n + c + hh) / den
    return o + s * d if s > 0 else None


def on_range(o, d, r):
    a = d[0] ** 2 + d[1] ** 2
    b = 2 * (o[0] * d[0] + o[1] * d[1])
    c = o[0] ** 2 + o[1] ** 2 - r ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = (-b + np.sqrt(disc)) / (2 * a)
    return o + s * d if s > 0 else None


def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


_GT_CACHE = {}


def gt_yaw(clip, fi):
    """训练/诊断用的 GT 航向：运动目标用 GT 轨迹方向（位移 >0.5 m），否则用文本 azimuth（第 14 帧的 rotation）。"""
    fr = clip["frames"]
    a, b = max(0, fi - 3), min(29, fi + 3)
    if fr[a]["xyz"] is not None and fr[b]["xyz"] is not None:
        dv = fr[b]["xyz"][:2] - fr[a]["xyz"][:2]
        if np.hypot(dv[0], dv[1]) > 0.5:
            return float(np.arctan2(dv[1], dv[0]))
    az = parse_text(clip["desc"])["azimuth"]
    return float(np.radians(az)) if az is not None else 0.0


def gt_box(clip, fi):
    """GT 目标在第 fi 帧对应的检测框。center：含 GT 中心投影的最小框（v11 及以前）；
    rect：GT 3D 框（GT 尺寸 + GT 航向）投影矩形（裁到图内）IoU 最大且 >=0.3 的框，失败再退回 center。
    ana_match.py：center 方式在 0029/0049/0058/0059/0060/0061 等序列有 13% 的帧匹配到邻车/小框。"""
    key = (clip["split"], clip["vid"], fi)
    if key in _GT_CACHE:
        return _GT_CACHE[key]
    f = clip["frames"][fi]
    res = None
    if f["xyz"] is not None:
        calib = load_calib(f["img"])
        boxes = DETS.get(f"{clip['split']}/{f['img']}", [])
        if args.gt_match == "rect" and boxes:
            R, ok = bf_rects(f["xyz"][None], f["lwh"], gt_yaw(clip, fi), calib)
            r = [max(0.0, R[0, 0]), max(0.0, R[0, 1]), min(W_IMG, R[0, 2]), min(H_IMG, R[0, 3])]
            if ok[0] and r[2] > r[0] and r[3] > r[1]:
                best = max(boxes, key=lambda b: iou2d(b, r))
                if iou2d(best, r) >= 0.3:
                    res = best
        if res is None:
            uv, _ = project(f["xyz"][None], calib)
            hits = [b for b in boxes if contains(b, uv[0])]
            res = min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if hits else None
    _GT_CACHE[key] = res
    return res


# ---------------------------------------------------------------- 地平面
def fit_ground(clips, exclude_tracks=frozenset()):
    """exclude_tracks: {(seq, track)}，这些目标不参与单相机高度偏移估计（防止同一物体的标签传到测试）。"""
    rows = collections.defaultdict(list)
    track_rows = collections.defaultdict(list)
    for c in clips:
        if (c["seq"], c["track"]) in exclude_tracks:
            continue
        for f in c["frames"]:
            rows[c["seq"]].append([f["xyz"][0], f["xyz"][1], f["xyz"][2] - f["lwh"][2] / 2])
            track_rows[c["seq"]].append(c["track"])
    allr = np.concatenate([np.array(v) for v in rows.values()])
    A = np.c_[allr[:, 0], allr[:, 1], np.ones(len(allr))]
    g = np.linalg.lstsq(A, allr[:, 2], rcond=None)[0]
    planes = {"_global": g}
    info = {}
    for seq, r in rows.items():
        r = np.array(r)
        A = np.c_[r[:, 0], r[:, 1], np.ones(len(r))]
        if args.ground == "global":
            planes[seq] = g
        elif args.ground == "global_offset":
            off = np.median(r[:, 2] - A @ g)
            planes[seq] = g + np.array([0, 0, off])
        elif args.ground == "track_offset":
            res = r[:, 2] - A @ g
            trk = np.array(track_rows[seq])
            meds, wts = [], []
            for tid in np.unique(trk):
                v = res[trk == tid]
                meds.append(np.median(v))
                wts.append(1.0 / (np.var(v) + 0.05 ** 2))
            off = float(np.sum(np.array(meds) * np.array(wts)) / np.sum(wts))
            planes[seq] = g + np.array([0, 0, off])
        else:
            planes[seq] = np.linalg.lstsq(A, r[:, 2], rcond=None)[0]
        info[seq] = (np.ptp(r[:, 0]), np.ptp(r[:, 1]), planes[seq])
    return planes, info


def fit_offset(clips):
    offs = []
    for c in clips:
        for fi in range(0, 30, 3):
            f = c["frames"][fi]
            uv, _ = project(f["xyz"][None], load_calib(f["img"]))
            b = gt_box(c, fi)
            if b is not None and not (args.gt_match == "rect" and trunc_flags(b, args.bf_margin).any()):
                offs.append([(uv[0, 0] - (b[0] + b[2]) / 2) / (b[2] - b[0]), (uv[0, 1] - (b[1] + b[3]) / 2) / (b[3] - b[1])])
    return np.median(np.array(offs), 0)


def fit_bf_bias(clips):
    """boxfit 各边偏置/尺度：GT 中心 + 文本尺寸 + 文本航向 的投影矩形 与 GT 对应检测框（未截断）的差 / 框尺寸。"""
    res = []
    for c in clips:
        t = parse_text(c["desc"])
        if t["lwh"] is None:
            continue
        yaw = np.radians(t["azimuth"]) if t["azimuth"] is not None else 0.0
        for fi in range(0, 30, 3):
            f = c["frames"][fi]
            b = gt_box(c, fi)
            if b is None or trunc_flags(b, args.bf_margin).any():
                continue
            R, ok = bf_rects(f["xyz"][None], t["lwh"], gt_yaw(c, fi) if "mov" in (c["state"] or "").lower() else yaw,
                             load_calib(f["img"]))
            if ok[0]:
                S = np.array([b[2] - b[0], b[3] - b[1], b[2] - b[0], b[3] - b[1]])
                res.append((R[0] - np.array(b[:4])) / S)
    res = np.array(res)
    bias = np.median(res, 0)
    sig = 1.4826 * np.median(np.abs(res - bias), 0)
    return bias, sig


# ---------------------------------------------------------------- 候选
def range_curve(dist):
    if len(dist) >= 3 and args.range_interp == "quad":
        return np.polyval(np.polyfit([0, args.anchor_mid, 29], dist[:3], 2), np.arange(30))
    if len(dist) >= 3:
        return np.interp(np.arange(30), [0, args.anchor_mid, 29], dist[:3])
    if len(dist) == 2:
        return np.interp(np.arange(30), [0, 29], dist[:2])
    if len(dist) == 1:
        return np.full(30, dist[0])
    return None


def clip_candidates(clip, planes, offset, color_model):
    t = parse_text(clip["desc"])
    chunks = parse_chunks(clip["desc"])
    lwh = t["lwh"] if t["lwh"] is not None else np.array([4.5, 1.8, 1.5])
    plane = planes.get(clip["seq"], planes["_global"])
    rc = range_curve(t["distances"])
    allowed = TYPE_TO_COCO.get(clip["type"], {2, 5, 7})
    cname = color_of(clip)
    frames = []
    for fi in range(30):
        f = clip["frames"][fi]
        calib = load_calib(f["img"])
        boxes = DETS.get(f"{clip['split']}/{f['img']}", [])
        color_frame = fi in ANCHORS or (args.color_every > 0 and fi % args.color_every == 0)
        key = f"{clip['split']}/{f['img']}"
        clip_probs = CLIP["probs"].get(key) if CLIP is not None else None
        efn_probs = EFN["probs"].get(key) if EFN is not None else None
        img = None
        if CLIP is None and EFN is None and color_frame and boxes and color_model is not None:
            img = cv2.imread(image_path(clip, fi))
        cands = []
        for bi, b in enumerate(boxes):
            o, d = ray_center(b, calib, offset)
            p = on_plane(o, d, plane, lwh[2] / 2)
            if p is None:
                continue
            r_plane = np.hypot(p[0], p[1])
            feat = np.zeros(N_FEAT)
            if rc is not None:
                rel_err = abs(r_plane - rc[fi]) / max(rc[fi], 1.0)
                at_edge = b[0] < 12 or b[1] < 12 or b[2] > W_IMG - 12 or b[3] > H_IMG - 12
                tol = args.range_tol_trunc if (at_edge and args.range_tol_trunc >= 0) else args.range_tol
                feat[0] = min(max(rel_err - tol, 0.0), 1.0)
            feat[1] = 0.0 if int(b[5]) in allowed else 1.0
            feat[2] = b[4]
            row = min(2, int(((b[1] + b[3]) / 2) / (H_IMG / 3)))
            col = min(2, int(((b[0] + b[2]) / 2) / (W_IMG / 3)))
            if img is not None:
                feat[5] = -np.log(color_model.prob(color_feature(img, b), cname))
            elif color_frame and cname != "other" and (clip_probs is not None or efn_probs is not None):
                vals = []
                if clip_probs is not None and cname in CLIP_INDEX:
                    vals.append(-np.log(max(float(clip_probs[bi][CLIP_INDEX[cname]]), args.clip_floor, 1e-4)))
                if efn_probs is not None and cname in EFN_GROUPS:
                    pe = efn_probs[bi]
                    if np.isfinite(pe).all():
                        s = float(sum(pe[EFN_INDEX[g]] for g in EFN_GROUPS[cname]))
                        vals.append(-np.log(max(s, args.clip_floor, 1e-4)))
                if vals:
                    feat[5] = float(np.mean(vals))
            if fi in ANCHORS:
                ch = chunks[ANCHORS[fi]]
                if ch["region"] is not None:
                    feat[3] = abs(row - ch["region"][0]) + abs(col - ch["region"][1])
                if ch["trunc"] is not None:
                    edge = b[0] < 4 or b[1] < 4 or b[2] > W_IMG - 4 or b[3] > H_IMG - 4
                    feat[4] = float(edge != bool(ch["trunc"]))
                feat[6] = abs(np.log(max(b[3] - b[1], 1) / max(b[2] - b[0], 1)))
                feat[7] = 1.0
            p_rng = on_range(o, d, rc[fi]) if rc is not None else None
            cands.append({"box": b, "o": o, "d": d, "p_plane": p, "feat": feat, "row": row, "col": col,
                          "r_plane": r_plane, "cy": (b[1] + b[3]) / 2, "fi": fi,
                          "P": p_rng if p_rng is not None else p})
        if fi in ANCHORS and chunks[ANCHORS[fi]]["ordinal"] is not None:
            num, direction = chunks[ANCHORS[fi]]["ordinal"]
            if args.ordinal_mode == "gen":
                # 生成脚本的名次：同格车辆里，按格子左上角到框底边中点的距离排序
                rp_by_id = {id(cd["box"]): cd["r_plane"] for cd in cands}
                ref = [b for b in boxes if int(b[5]) in (2, 5, 7) and b[4] >= args.ord_conf
                       and rp_by_id.get(id(b), 1e9) <= args.ord_rmax]
                ref = [ref[i] for i in nms_agnostic(ref)]
                for cd in cands:
                    b = cd["box"]
                    cell = gen_cell(b)
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    corner = (W_IMG / 3 * math.floor(cx / (W_IMG / 3)), H_IMG / 3 * math.floor(cy / (H_IMG / 3)))
                    d_self = math.hypot(cx - corner[0], b[3] - corner[1])
                    rank = 1 + sum(1 for r in ref if gen_cell(r) == cell and iou2d(r, b) < 0.7 and
                                   math.hypot((r[0] + r[2]) / 2 - corner[0], r[3] - corner[1]) < d_self)
                    cd["ord"] = rank
                    cd["feat"][8] = min(abs(rank - num), 3) / 3.0
            else:
                groups = collections.defaultdict(list)
                for cd in cands:
                    groups[(cd["row"], cd["col"])].append(cd)
                for g in groups.values():
                    key = (lambda cd: cd["cy"]) if direction == "top" else (lambda cd: cd["r_plane"])
                    for rank, cd in enumerate(sorted(g, key=key), 1):
                        cd["feat"][8] = min(abs(rank - num), 3) / 3.0
        ch_a = chunks[ANCHORS[fi]] if fi in ANCHORS else None
        need_rel = args.rel_feat and ch_a is not None and ch_a["rel"] is not None
        need_nc = args.near_color_feat and ch_a is not None and ch_a["near_color"] is not None
        if (need_rel or need_nc) and cands:
            ref_idx = [i for i, b in enumerate(boxes) if int(b[5]) in (2, 5, 7) and b[4] >= args.ord_conf]
            ref_idx = [ref_idx[k] for k in nms_agnostic([boxes[i] for i in ref_idx])]
            ref_xy = {}
            for i in ref_idx:
                o_r, d_r = ray_center(boxes[i], calib, offset)
                p_r = on_plane(o_r, d_r, plane, 0.75)
                if p_r is not None and math.hypot(p_r[0], p_r[1]) <= args.ord_rmax:
                    ref_xy[i] = p_r[:2]
            for cd in cands:
                sxy = cd["P"][:2]
                best, near_i = 1e9, None
                for i, rxy in ref_xy.items():
                    if iou2d(boxes[i], cd["box"]) >= 0.7:
                        continue
                    dd = math.hypot(rxy[0] - sxy[0], rxy[1] - sxy[1])
                    if dd < best:
                        best, near_i = dd, i
                if near_i is None:
                    continue
                if need_rel:
                    want = REL_VEC[ch_a["rel"]]
                    rel = space_relate3d(sxy, ref_xy[near_i])
                    got = REL_VEC[rel]
                    cd["rel"] = rel
                    cd["feat"][9] = (abs(got[0] - want[0]) + abs(got[1] - want[1])) / 4.0
                if need_nc:
                    nc = ch_a["near_color"]
                    vals = []
                    if clip_probs is not None and nc in CLIP_INDEX:
                        vals.append(-np.log(max(float(clip_probs[near_i][CLIP_INDEX[nc]]), 1e-3)))
                    if efn_probs is not None and nc in EFN_GROUPS and np.isfinite(efn_probs[near_i]).all():
                        s = float(sum(efn_probs[near_i][EFN_INDEX[g]] for g in EFN_GROUPS[nc]))
                        vals.append(-np.log(max(s, 1e-3)))
                    if vals:
                        cd["feat"][10] = float(np.mean(vals))
        frames.append(cands)
    state = (clip.get("state") or "").lower()
    heading = frame_headings(t, chunks, state)
    # boxfit 用的逐帧航向：运动目标用文本航向插值（frame_headings），停车目标用文本 azimuth / 首段朝向词
    if heading is not None:
        yaws = np.arctan2(heading[:, 1], heading[:, 0])
    else:
        deg = t["azimuth"] if t["azimuth"] is not None else next(
            (ch["orient"] for ch in chunks if ch["orient"] is not None), 0.0)
        yaws = np.full(30, np.radians(deg))
    return {"t": t, "lwh": lwh, "plane": plane, "rc": rc, "frames": frames, "state": state,
            "heading": heading, "yaws": yaws, "imgs": [f["img"] for f in clip["frames"]],
            "seq": clip["seq"], "cam": cam_key(clip["frames"][0]["img"])}


def frame_headings(t, chunks, state):
    """运动目标每帧的期望航向单位向量 (30,2)；第 0 段优先用文本数值 azimuth，其余段用朝向词，关键帧之间插值。"""
    if "mov" not in state:
        return None
    degs = [t["azimuth"] if t["azimuth"] is not None else chunks[0]["orient"], chunks[1]["orient"], chunks[2]["orient"]]
    if args.heading_mode == "first":
        degs = [degs[0], None, None]
    if degs[0] is None:
        degs[0] = degs[1] if degs[1] is not None else degs[2]
    if degs[0] is None:
        return None
    for k in (1, 2):
        if degs[k] is None:
            degs[k] = degs[k - 1]
    us = [np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in degs]
    keys = [0, args.anchor_mid, 29]
    out = np.zeros((30, 2))
    for fi in range(30):
        seg = 0 if fi <= keys[1] else 1
        a0, a1 = keys[seg], keys[seg + 1]
        alpha = (fi - a0) / max(a1 - a0, 1)
        v = (1 - alpha) * us[seg] + alpha * us[seg + 1]
        n = np.linalg.norm(v)
        out[fi] = v / n if n > 0.3 else (us[seg] if alpha < 0.5 else us[seg + 1])
    return out


# ---------------------------------------------------------------- 打分器（向量化 softmax 排序）
def train_scorer(samples, l2=1e-2, iters=400, lr=0.5):
    X = np.concatenate([s[0] for s in samples])
    sizes = np.array([len(s[0]) for s in samples])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    yidx = starts + np.array([s[1] for s in samples])
    seg = np.repeat(np.arange(len(samples)), sizes)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        s = -X @ w
        s = s - np.maximum.reduceat(s, starts)[seg]
        e = np.exp(s)
        p = e / np.add.reduceat(e, starts)[seg]
        px = np.add.reduceat(p[:, None] * X, starts)
        g = (X[yidx] - px).mean(0) + l2 * w
        w -= lr * g
    return w


def build_color_model(clips):
    feats, labels, keys = [], [], []
    for c in clips:
        name = color_of(c)
        for fi in (0, 10, 20, 29):
            b = gt_box(c, fi)
            if b is None:
                continue
            feats.append(color_feature(cv2.imread(image_path(c, fi)), b))
            labels.append(name)
            keys.append((c["seq"], c["track"]))
    return ColorKNN(feats, labels, keys=keys)


# ---------------------------------------------------------------- 跟踪与输出
def track_clip(cc, w):
    frames = cc["frames"]
    unary = []
    for fi, cands in enumerate(frames):
        if not cands:
            unary.append(np.array([]))
            continue
        X = np.array([cd["feat"] for cd in cands])
        if fi in ANCHORS:
            unary.append(args.anchor_w * (X @ w))
        else:
            # 非关键帧：距离 + 类别 + 颜色（颜色只在计算过的帧非零）
            unary.append(args.other_w * (w[0] * X[:, 0] + w[1] * X[:, 1] + w[5] * X[:, 5]))

    heading = cc.get("heading")

    def pair_cost(p, c):
        iou = iou2d(p["box"], c["box"])
        cost = args.iou_w * (1 - iou) if iou >= args.stay_iou else args.switch_cost
        if args.dir_w > 0 and heading is not None:
            delta = c["P"][:2] - p["P"][:2]
            n = np.hypot(delta[0], delta[1])
            if n > args.dir_min:
                cost += args.dir_w * (1 - (delta @ heading[c["fi"]]) / n) / 2
        return cost

    if any(len(u) == 0 for u in unary):
        track, prev = [], None
        for fi, cands in enumerate(frames):
            if not cands:
                track.append(prev)
                continue
            cost = unary[fi].copy()
            if prev is not None:
                cost += np.array([pair_cost(prev, cd) for cd in cands])
            prev = cands[int(np.argmin(cost))]
            track.append(prev)
        return track
    acc, back = unary[0].copy(), []
    for fi in range(1, 30):
        pair = np.array([[pair_cost(p, c) for p in frames[fi - 1]] for c in frames[fi]])
        tot = pair + acc[None, :]
        idx = tot.argmin(1)
        acc = tot[np.arange(len(idx)), idx] + unary[fi]
        back.append(idx)
    j = int(acc.argmin())
    path = [j]
    for idx in reversed(back):
        j = int(idx[j])
        path.append(j)
    path.reverse()
    return [frames[fi][path[fi]] for fi in range(30)]


def oracle_track(clip, cc):
    track = []
    for fi in range(30):
        b = gt_box(clip, fi)
        hit = None
        if b is not None:
            for cd in cc["frames"][fi]:
                if cd["box"] is b or list(cd["box"]) == list(b):
                    hit = cd
                    break
        track.append(hit)
    return track


BF = {"bias": np.zeros(4), "sig": np.array([0.07, 0.045, 0.055, 0.05])}  # main() 里用 TRAIN 重新拟合


def bf_center(cc, cd, fi):
    """boxfit 中心（见 boxfit.py）；不适用时返回 None，由 ray 方式兜底。"""
    rc = cc["rc"]
    if rc is None or args.center_mode == "ray":
        return None
    b = cd["box"]
    tr = trunc_flags(b, args.bf_margin)
    if args.center_mode == "boxfit_trunc" and not tr.any():
        return None
    if args.center_mode == "boxfit_near" and not tr.any() and rc[fi] >= args.bf_near:
        return None
    if args.center_mode == "boxfit_far" and not tr.any() and rc[fi] < args.bf_near:
        return None
    p_ray = on_range(cd["o"], cd["d"], rc[fi])
    if p_ray is None:
        return None
    lwh = cc["lwh"]
    cen, _, _ = bf_fit_center(b, tr, lwh, cc["yaws"][fi], rc[fi], load_calib(cc["imgs"][fi]), cc["plane"],
                              hc=lwh[2] / 2 + args.bf_hc, zsig=args.bf_zsig, bias=BF["bias"], sig=BF["sig"],
                              theta0=float(np.arctan2(p_ray[1], p_ray[0])))
    return cen


def local_linear(v, k):
    """逐点局部线性回归平滑（窗口 ±k，边界处窗口单侧截断；对线性趋势无偏，避免滑动平均在首尾的偏差）。"""
    ts = np.arange(len(v))
    out = np.empty(len(v))
    for i in range(len(v)):
        lo, hi = max(0, i - k), min(len(v), i + k + 1)
        A = np.c_[ts[lo:hi] - i, np.ones(hi - lo)]
        out[i] = np.linalg.lstsq(A, v[lo:hi], rcond=None)[0][1]
    return out


def frame_center(cc, cd, fi):
    """单帧中心估计（未平滑）：boxfit 适用时用 boxfit，否则 ray 与文本距离圆柱求交 + z 混合。
    --blend_ray a>0：未截断的帧取 (1-a)*boxfit + a*ray（两种估计的误差不完全相关，平均可以降方差）。"""
    cen = bf_center(cc, cd, fi)
    if cen is not None:
        cen = np.asarray(cen, dtype=np.float64)
        if args.blend_ray > 0 and not trunc_flags(cd["box"], args.bf_margin).any():
            ray = ray_frame_center(cc, cd, fi)
            # 只混合横向和高度；径向两者都等于文本距离
            cen = (1 - args.blend_ray) * cen + args.blend_ray * ray
        return cen
    return ray_frame_center(cc, cd, fi)


def ray_frame_center(cc, cd, fi):
    lwh, plane, rc = cc["lwh"], cc["plane"], cc["rc"]
    p = on_range(cd["o"], cd["d"], rc[fi]) if rc is not None else None
    if p is None:
        p = cd["p_plane"]
    z_plane = plane[0] * p[0] + plane[1] * p[1] + plane[2] + lwh[2] / 2
    b = cd["box"]
    if args.trunc_z and (b[1] < 12 or b[3] > H_IMG - 12):
        z = z_plane  # 上下被图像边界截断：框中心不是物体中心，射线高度偏高
    elif args.zray_w >= 0:
        z = args.zray_w * p[2] + (1 - args.zray_w) * z_plane
    else:
        z = {"plane": z_plane, "ray": p[2], "avg": 0.5 * (z_plane + p[2])}[args.zsrc]
    return np.array([p[0], p[1], z])


# 按 sequence / 相机学到的中心系统偏差（横向、高度，单位米），main() 里用 TRAIN 片段填充
SEQ_BIAS, CAM_BIAS = {}, {}


def cam_key(img):
    return tuple(np.round(load_calib(img)["t"], 1))


def lateral_unit(p):
    r = math.hypot(p[0], p[1])
    return np.array([-p[1] / r, p[0] / r]) if r > 1e-6 else np.array([0.0, 1.0])


def apply_bias(cc, p):
    """p: 单帧中心；减去该片段所在 sequence（无则相机）的系统偏差（横向、高度）。"""
    if args.seq_bias <= 0:
        return p
    b = None
    if args.bias_level in ("seq", "seqcam"):
        b = SEQ_BIAS.get(cc["seq"])
    if b is None and args.bias_level in ("cam", "seqcam"):
        b = CAM_BIAS.get(cc["cam"])
    if b is None:
        return p
    lat, dz, n = b
    wgt = n / (n + args.seq_bias)
    q = p.copy()
    q[:2] -= wgt * lat * lateral_unit(p)
    q[2] -= wgt * dz
    return q


def fit_center_bias(pairs):
    """pairs: [(TRAIN 片段, 其候选 cc)]。用 GT 对应的检测框估计中心估计器的系统偏差：
    每段取逐帧误差（横向、高度）的中位数，再按 sequence / 相机平均。只用 TRAIN 标签。"""
    per_seq, per_cam = collections.defaultdict(list), collections.defaultdict(list)
    for c, cc in pairs:
        errs = []
        for fi in range(0, 30, 2):
            b = gt_box(c, fi)
            g = c["frames"][fi]["xyz"]
            if b is None or g is None:
                continue
            cd = next((x for x in cc["frames"][fi] if list(x["box"]) == list(b)), None)
            if cd is None:
                continue
            p = frame_center(cc, cd, fi)
            e = p - g
            errs.append([e[:2] @ lateral_unit(g), e[2]])
        if len(errs) >= 3:
            m = np.median(np.array(errs), 0)
            per_seq[c["seq"]].append(m)
            per_cam[cc["cam"]].append(m)
    for store, src in ((SEQ_BIAS, per_seq), (CAM_BIAS, per_cam)):
        store.clear()
        for k, v in src.items():
            v = np.array(v)
            store[k] = (float(np.mean(v[:, 0])), float(np.mean(v[:, 1])), len(v))


def to_output(cc, track, _second=False):
    out = np.zeros((30, 6))
    lwh, plane, rc = cc["lwh"], cc["plane"], cc["rc"]
    last = None
    for fi, cd in enumerate(track):
        out[fi, :3] = lwh
        if cd is None:
            cd = last
        if cd is None:
            out[fi, 3:] = [rc[fi] if rc is not None else 30.0, 0.0, plane[2] + lwh[2] / 2]
            continue
        last = cd
        out[fi, 3:] = apply_bias(cc, frame_center(cc, cd, fi))
    if args.zmode == "median":
        out[:, 5] = np.median(out[:, 5])
    parked = "park" in cc["state"]
    if args.smooth == "state":
        if parked:
            out[:, 3:] = np.median(out[:, 3:], axis=0)
        else:
            ts = np.arange(30)
            for k in range(3, 6):
                out[:, k] = np.polyval(np.polyfit(ts, out[:, k], 2), ts)
    elif args.smooth == "state_ma":
        if parked:
            out[:, 3:] = np.median(out[:, 3:], axis=0)
        else:
            for k in (3, 4):
                v = out[:, k].copy()
                out[:, k] = [np.mean(v[max(0, i - 2):i + 3]) for i in range(30)]
    elif args.smooth == "state_lin":
        # 运动目标：距离保持文本曲线（关键帧处精确），只对方位角做局部线性平滑。
        # ana_move.py：v11c 的 5 帧滑动平均在第 0/29 帧造成 ~0.45 m 的径向偏差。
        if parked:
            out[:, 3:] = np.median(out[:, 3:], axis=0)
        else:
            r = np.hypot(out[:, 3], out[:, 4])
            th = local_linear(np.unwrap(np.arctan2(out[:, 4], out[:, 3])), args.smooth_k)
            out[:, 3], out[:, 4] = r * np.cos(th), r * np.sin(th)
    if args.yaw_motion and not _second and "park" not in cc["state"] and args.center_mode != "ray":
        # 运动目标：用第一遍轨迹的运动方向代替文本航向（文本只有第 14 帧的 rotation 和朝向词），再做一遍 boxfit
        yaws = cc["yaws"].copy()
        for fi in range(30):
            a, b = max(0, fi - 3), min(29, fi + 3)
            dv = out[b, 3:5] - out[a, 3:5]
            if np.hypot(dv[0], dv[1]) > args.yaw_motion_min:
                yaws[fi] = math.atan2(dv[1], dv[0])
        cc2 = dict(cc)
        cc2["yaws"] = yaws
        return to_output(cc2, track, _second=True)
    return out


# ---------------------------------------------------------------- 划分
def make_split(train_all):
    if args.mode == "val":
        val_ids = set(read_ids("val"))
        return [c for c in train_all if c["vid"] not in val_ids], [c for c in train_all if c["vid"] in val_ids]
    if args.mode == "cv":
        keys = sorted({(c["seq"], c["track"]) for c in train_all})
        rng = np.random.RandomState(12345)
        perm = rng.permutation(len(keys))
        held = {keys[i] for j, i in enumerate(perm) if j % args.nfold == args.fold}
        return ([c for c in train_all if (c["seq"], c["track"]) not in held],
                [c for c in train_all if (c["seq"], c["track"]) in held])
    if args.mode == "seen" and args.seen_split == "clip":
        by_seq = collections.defaultdict(list)
        for c in train_all:
            by_seq[c["seq"]].append(c)
        fit, ev = [], []
        for seq, cs in by_seq.items():
            cs = sorted(cs, key=lambda c: c["vid"])
            for i, c in enumerate(cs):
                (ev if (len(cs) >= 3 and i % 3 == 1) else fit).append(c)
        return fit, ev
    if args.mode == "seen":
        # 按 (sequence, track) 分组留出：同一物体的所有片段要么全在 fit，要么全在 eval
        by_seq = collections.defaultdict(lambda: collections.defaultdict(list))
        for c in train_all:
            by_seq[c["seq"]][c["track"]].append(c)
        fit, ev = [], []
        for seq, tracks in by_seq.items():
            keys = sorted(tracks)
            for i, k in enumerate(keys):
                (ev if (len(keys) >= 3 and i % 3 == 1) else fit).extend(tracks[k])
        return fit, ev
    return train_all, load_clips("test")


# 参数文件里记录的配置项：除了下面这些与输入输出路径、运行模式有关的项，其余都会影响拟合结果
_RUNTIME_KEYS = {"mode", "fold", "dets", "efn_cache", "clip_cache", "dump", "dump_params", "load_params", "out", "tag"}


def load_params(path):
    """读取 --dump_params 写出的参数（全部在 TRAIN 上拟合），返回 planes, offset, w。"""
    P = json.load(open(path, encoding="utf-8"))
    diff = {k: (v, getattr(args, k, None)) for k, v in P["config"].items()
            if k not in _RUNTIME_KEYS and getattr(args, k, None) != v}
    if diff:
        print("WARNING: 当前命令行与参数文件的配置不同（参数文件值, 当前值）：", diff)
    planes = {k: np.array(v, dtype=np.float64) for k, v in P["ground_planes"].items()}
    offset = np.array(P["center_offset"], dtype=np.float64)
    BF["sig"] = np.array(P["boxfit_side_sigma"], dtype=np.float64)
    if args.bf_bias:
        BF["bias"] = np.array(P["boxfit_side_bias"], dtype=np.float64)
    w = np.array([P["scorer_weights"][name] for name in FEAT_NAMES], dtype=np.float64)
    print(f"loaded params from {path}: fitted on {P['n_fit_clips']} TRAIN clips, "
          f"{P['n_scorer_samples']} scorer samples; offset={offset.round(3)} sig={BF['sig'].round(3)}")
    print("weights", dict(zip(FEAT_NAMES, w.round(3))))
    return planes, offset, w


def save_params(path, fit, n_samples, planes, offset, bb, w):
    P = {"method": "Tenn-VLSOT-Geo（望远队，2026 VLP 挑战赛赛道 A）",
         "note": "全部参数只用 TRAIN 标签拟合；用法：pipeline3.py --mode test <同一配置> --load_params 本文件",
         "n_fit_clips": len(fit), "n_scorer_samples": n_samples,
         "feat_names": FEAT_NAMES,
         "scorer_weights": {name: float(v) for name, v in zip(FEAT_NAMES, w)},
         "ground_planes": {k: [float(x) for x in v] for k, v in planes.items()},
         "center_offset": [float(x) for x in offset],
         "boxfit_side_bias": [float(x) for x in bb] if bb is not None else None,
         "boxfit_side_sigma": [float(x) for x in BF["sig"]],
         "config": {k: v for k, v in vars(args).items() if k not in _RUNTIME_KEYS}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(P, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("params written to", path)


def main():
    if args.load_params:
        if args.mode != "test":
            raise SystemExit("--load_params 只用于 --mode test")
        if args.seq_bias > 0 or (CLIP is None and EFN is None):
            raise SystemExit("--load_params 不支持 --seq_bias 与 knn 颜色模型（二者都要在推理时读 TRAIN 数据）")
        planes, offset, w = load_params(args.load_params)
        evals, color_model = load_clips("test"), None
        run_eval(evals, planes, offset, w, color_model, fit=[])
        return
    train_all = load_clips("train")
    fit, evals = make_split(train_all)
    if args.fit_frac < 1.0:
        rng = np.random.RandomState(args.fit_seed)
        keep = rng.permutation(len(fit))[:max(1, int(round(args.fit_frac * len(fit))))]
        fit = [fit[i] for i in sorted(keep)]
    test_clips = load_clips("test")
    test_seqs = {c["seq"] for c in test_clips}
    exclude = frozenset((c["seq"], c["track"]) for c in test_clips) if args.mode == "test" else frozenset()
    planes, info = fit_ground(fit, exclude)
    bb = None
    if exclude:
        print("excluded from ground offsets (same seq+track as a test target):",
              sorted({(c["seq"], c["track"]) for c in fit if (c["seq"], c["track"]) in exclude}))
    offset = fit_offset(fit)
    if args.center_mode != "ray":
        bb, ss = fit_bf_bias(fit)
        BF["sig"] = ss
        if args.bf_bias:
            BF["bias"] = bb
        print("boxfit side bias", bb.round(3), "sig", ss.round(3), "(bias used)" if args.bf_bias else "(bias not used)")
    print(f"center_mode={args.center_mode} gt_match={args.gt_match} smooth_k={args.smooth_k} "
          f"range_interp={args.range_interp} bf_zsig={args.bf_zsig} bf_hc={args.bf_hc} bf_near={args.bf_near}")
    print(f"mode={args.mode} ground={args.ground} zsrc={args.zsrc} smooth={args.smooth} zmode={args.zmode} "
          f"switch={args.switch_cost} stay_iou={args.stay_iou} color_every={args.color_every} oracle={args.oracle} "
          f"color_model={args.color_model} loo_track={args.loo_track} zray_w={args.zray_w} dir_w={args.dir_w} "
          f"heading={args.heading_mode} clip_floor={args.clip_floor} "
          f"fit={len(fit)} eval={len(evals)} offset={offset.round(3)}")
    for s in sorted(test_seqs):
        if s in info:
            xr, yr, pl = info[s]
            print(f"  seq {s}: x-span {xr:5.1f} y-span {yr:5.1f} plane {np.round(pl, 3)}")
    color_model = build_color_model(fit) if (CLIP is None and EFN is None) else None
    samples, fit_pairs = [], []
    for c in fit:
        cc = clip_candidates(c, planes, offset, color_model)
        if args.seq_bias > 0 and (c["seq"], c["track"]) not in exclude:
            fit_pairs.append((c, cc))  # 与测试目标同 (seq, track) 的训练片段不参与偏差估计
        for fi in range(30):
            cands = cc["frames"][fi]
            if len(cands) < 2:
                continue
            b = gt_box(c, fi)
            if b is None:
                continue
            ys = [i for i, cd in enumerate(cands) if list(cd["box"]) == list(b)]
            if ys:
                samples.append((np.array([cd["feat"] for cd in cands]), ys[0]))
    w = train_scorer(samples)
    print("weights", dict(zip(FEAT_NAMES, w.round(3))), "samples", len(samples))
    if args.seq_bias > 0:
        fit_center_bias(fit_pairs)
        del fit_pairs
        print("centre bias (lateral, dz, n) for test seqs:",
              {s: tuple(round(x, 3) for x in SEQ_BIAS[s]) for s in sorted(test_seqs) if s in SEQ_BIAS})
    if args.dump_params:
        save_params(args.dump_params, fit, len(samples), planes, offset, bb, w)
    run_eval(evals, planes, offset, w, color_model, fit, test_seqs)


def run_eval(evals, planes, offset, w, color_model, fit, test_seqs=frozenset()):
    """对 evals 片段做选择与定位；非 test 模式同时用 TRAIN 标签评估，test 模式写提交文件。"""
    preds, tracks = {}, {}
    for c in evals:
        planes_c, color_c = planes, color_model
        if args.loo_track and args.mode == "seen":
            planes_c, _ = fit_ground(fit, frozenset({(c["seq"], c["track"])}))
            if color_model is not None:
                color_c = color_model.without((c["seq"], c["track"]))
        cc = clip_candidates(c, planes_c, offset, color_c)
        tracks[c["vid"]] = oracle_track(c, cc) if args.oracle else track_clip(cc, w)
        preds[c["vid"]] = to_output(cc, tracks[c["vid"]])

    if args.mode != "test":
        gt = {c["vid"]: gt_array(c) for c in evals}
        print("ALL      ", len(gt), evaluate(preds, gt))
        print("PLATFORM ", len(gt), evaluate_platform(preds, gt))
        sub = {v: g for v, g in gt.items() if next(c for c in evals if c["vid"] == v)["seq"] in test_seqs}
        if sub:
            print("TEST-SEQS", len(sub), evaluate({v: preds[v] for v in sub}, sub))
            print("PLAT-SEQS", len(sub), evaluate_platform({v: preds[v] for v in sub}, sub))
        hit, bad = [], []
        for c in evals:
            h = []
            for fi in range(30):
                b, cd = gt_box(c, fi), tracks[c["vid"]][fi]
                h.append(b is not None and cd is not None and list(cd["box"]) == list(b))
            hit += h
            ez = np.abs(preds[c["vid"]][:, 5] - gt[c["vid"]][:, 5]).mean()
            if np.mean(h) < 0.5 or ez > 0.5:
                bad.append((c["vid"], c["seq"], round(float(np.mean(h)), 2), round(float(ez), 2)))
        print("selection hit %.3f" % np.mean(hit))
        print("bad clips (vid, seq, hit, |dz|):", bad)
        err = np.concatenate([np.abs(preds[v][:, 3:] - gt[v][:, 3:]) for v in gt])
        print("abs err x/y/z mean", err.mean(0).round(3), "median", np.median(err, 0).round(3))
        if args.dump:
            dump = {}
            for c in evals:
                v = c["vid"]
                per = evaluate_platform({v: preds[v]}, {v: gt[v]})
                dump[v] = {"seq": c["seq"], "track": c["track"], "type": c["type"], "color": c.get("color"),
                           "state": c.get("state"), "desc": c["desc"], "plat_sr50": per["SR@0.5"],
                           "true_sr50": evaluate({v: preds[v]}, {v: gt[v]})["SR@0.5"],
                           "pred": preds[v].round(3).tolist(), "gt": gt[v].round(3).tolist(),
                           "sel": [None if cd is None else [float(x) for x in cd["box"]] for cd in tracks[v]],
                           "gtbox": [None if b is None else [float(x) for x in b] for b in
                                     (gt_box(c, fi) for fi in range(30))],
                           "imgs": [f["img"] for f in c["frames"]]}
            json.dump(dump, open(args.dump, "w"))
            print("dumped", len(dump), "clips to", args.dump)
    else:
        out_dir = f"{args.out}/test_{args.tag or 'v3'}"
        os.makedirs(out_dir + "/result", exist_ok=True)
        for vid, p in preds.items():
            with open(f"{out_dir}/result/{vid}.txt", "w") as fh:
                for i, row in enumerate(p):
                    fh.write("%d %s\n" % (i + 1, " ".join("%.4f" % v for v in row)))
        with zipfile.ZipFile(f"{out_dir}/result.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for vid in sorted(preds):
                z.write(f"{out_dir}/result/{vid}.txt", f"result/{vid}.txt")
        json.dump({vid: [None if cd is None else cd["box"] for cd in tr] for vid, tr in tracks.items()},
                  open(f"{out_dir}/tracks.json", "w"))
        for vid in sorted(preds):
            p = preds[vid]
            tr = tracks[vid]
            switches = sum(1 for t in range(1, 30) if tr[t] is not None and tr[t - 1] is not None
                           and iou2d(tr[t - 1]["box"], tr[t]["box"]) < args.stay_iou)
            print(vid, "xyz f0", p[0, 3:].round(2), "f29", p[29, 3:].round(2),
                  "bottom z", (p[:, 5] - p[:, 2] / 2).mean().round(2), "switches", switches)
        print("written", out_dir)


if __name__ == "__main__":
    main()
