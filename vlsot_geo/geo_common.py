"""VLSOT 几何流水线公共模块：数据读取、标定、投影、文本解析、评测指标。

坐标约定（来自官方 TRAIN 标签实测）：每帧标签为 [img, l, w, h, x, y, z]，
xyz 位于 V2X-Seq 路侧 virtuallidar 坐标系；文本中的距离 = sqrt(x^2 + y^2)。

路径用环境变量配置（见 README“数据准备”）：
  VLSOT_DATA   官方数据集解压目录，内含 train/ test/，各自有 json/ 与 img/
  VLSOT_CALIB  V2X-Seq-SPD 路侧标定目录（infrastructure-side/calib），内含 camera_intrinsic/ 与 virtuallidar_to_camera/
  VLSOT_WORK   工作目录，存放 weights/ cache/ out/（默认为仓库根目录）
  VLSOT_SPLIT  可选，只有 --mode val 需要（val.txt 片段列表）
"""
import glob
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("VLSOT_DATA", os.path.join(REPO, "data", "dataset"))
CAL = os.environ.get("VLSOT_CALIB", os.path.join(REPO, "data", "calib"))
WORK = os.environ.get("VLSOT_WORK", REPO)
SPLIT = os.environ.get("VLSOT_SPLIT", os.path.join(REPO, "data", "split"))


# ---------------------------------------------------------------- 数据
def load_clips(split):
    """split: 'train' 或 'test'。返回按 videoID 排序的片段列表。
    TEST 的 json 里每帧只有图像名（没有标签），此时 lwh / xyz 为 None。"""
    clips = []
    for path in sorted(glob.glob(f"{DATA}/{split}/json/*.json")):
        item = json.load(open(path))[0]
        frames = []
        for i in range(30):
            value = item[f"frame{i}"]
            if isinstance(value, str):
                frames.append({"img": value, "lwh": None, "xyz": None})
            else:
                frames.append({"img": value[0],
                               "lwh": np.array(value[1:4], dtype=np.float64),
                               "xyz": np.array(value[4:7], dtype=np.float64)})
        clips.append({"vid": item["videoID"], "seq": item["sequence_id"],
                      "track": item["track_id"], "type": item.get("type", ""),
                      "color": item.get("color", ""), "state": item.get("state", ""),
                      "desc": item["description"], "frames": frames, "split": split})
    return clips


def image_path(clip, frame_index):
    """片段第 frame_index 帧的图像路径：$VLSOT_DATA/<split>/img/<图像名>。"""
    return f"{DATA}/{clip['split']}/img/{clip['frames'][frame_index]['img']}"


def read_ids(name):
    """读取 $VLSOT_SPLIT/<name>.txt 中的片段 ID 列表（只有 --mode val 用到）。"""
    return [x.strip() for x in open(f"{SPLIT}/{name}.txt") if x.strip()]


# ---------------------------------------------------------------- 标定
_CALIB_CACHE = {}


def load_calib(img_name):
    """读取图像对应的 V2X-Seq-SPD 路侧标定（与图像同名的 json，结果缓存）。
    返回 K（3×3 内参）、D（畸变系数 k1 k2 p1 p2 k3）、R 与 t（virtuallidar → 相机的旋转与平移）。"""
    stem = os.path.splitext(img_name)[0]
    if stem in _CALIB_CACHE:
        return _CALIB_CACHE[stem]
    intr = json.load(open(f"{CAL}/camera_intrinsic/{stem}.json"))
    ext = json.load(open(f"{CAL}/virtuallidar_to_camera/{stem}.json"))
    calib = {"K": np.array(intr["cam_K"], dtype=np.float64).reshape(3, 3),
             "D": np.array(intr["cam_D"], dtype=np.float64),
             "R": np.array(ext["rotation"], dtype=np.float64),
             "t": np.array(ext["translation"], dtype=np.float64).reshape(3)}
    _CALIB_CACHE[stem] = calib
    return calib


def lidar_to_cam(points, calib):
    """virtuallidar 坐标 (N, 3) → 相机坐标 (N, 3)：p_cam = R p + t。"""
    points = np.atleast_2d(points)
    return points @ calib["R"].T + calib["t"]


def project(points, calib, distort=False):
    """virtuallidar 点 -> 像素 (N,2)，同时返回相机深度。"""
    cam = lidar_to_cam(points, calib)
    xn = cam[:, :2] / cam[:, 2:3]
    if distort:
        k1, k2, p1, p2, k3 = calib["D"]
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        xn = np.stack([xd, yd], axis=1)
    K = calib["K"]
    uv = np.stack([K[0, 0] * xn[:, 0] + K[0, 2], K[1, 1] * xn[:, 1] + K[1, 2]], axis=1)
    return uv, cam[:, 2]


def undistort_normalized(uv, calib, iters=20):
    """像素 → 去畸变后的归一化相机坐标 (N, 2)：对 k1 k2 p1 p2 k3 畸变模型做 iters 次定点迭代求逆。"""
    K = calib["K"]
    xd = (uv[:, 0] - K[0, 2]) / K[0, 0]
    yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
    k1, k2, p1, p2, k3 = calib["D"]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return np.stack([x, y], axis=1)


def pixel_rays(uv, calib, distort=False):
    """像素 -> virtuallidar 坐标系下的射线 (origin(3), dirs(N,3))。"""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    if distort:
        xn = undistort_normalized(uv, calib)
    else:
        K = calib["K"]
        xn = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]], axis=1)
    dirs_cam = np.concatenate([xn, np.ones((len(xn), 1))], axis=1)
    R, t = calib["R"], calib["t"]
    origin = np.linalg.solve(R, -t)
    dirs = np.linalg.solve(R, dirs_cam.T).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return origin, dirs


def intersect_plane(origin, dirs, plane):
    """plane=(a,b,c): z = a x + b y + c。返回交点 (N,3)。"""
    a, b, c = plane
    normal = np.array([a, b, -1.0])
    denom = dirs @ normal
    s = -(origin @ normal + c) / denom
    return origin + s[:, None] * dirs


# ---------------------------------------------------------------- 文本
_NUM = r"(\d+(?:\.\d+)?)"


def parse_text(desc):
    """返回 dims(l,w,h) 与按出现顺序的距离列表。"""
    text = desc.lower()
    dims = {}
    patterns = {
        "h": [_NUM + r"\s*meters?\s*(?:tall|in height|high)", r"height of\s*" + _NUM],
        "l": [_NUM + r"\s*meters?\s*(?:long|in length)", r"length of\s*" + _NUM],
        "w": [_NUM + r"\s*meters?\s*(?:wide|in width)", r"width of\s*" + _NUM],
    }
    spans = []
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                dims[key] = float(m.group(1))
                spans.append(m.span())
                break
    distances = []
    for m in re.finditer(_NUM + r"\s*(?:meters?|m)\b", text):
        if any(s[0] <= m.start() < s[1] for s in spans):
            continue
        tail = text[m.end():m.end() + 14]
        if re.match(r"\s*(tall|long|wide|in height|in length|in width|high)", tail):
            continue
        distances.append(float(m.group(1)))
    lwh = None
    if all(k in dims for k in "lwh"):
        lwh = np.array([dims["l"], dims["w"], dims["h"]])
    region = parse_region(text)
    azimuth = None
    m = re.search(r"azimuth of\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        azimuth = float(m.group(1))
    return {"lwh": lwh, "distances": distances, "azimuth": azimuth, "region": region}


def parse_region(text):
    """解析首句中的画面区域，返回 (row, col)，取值 0/1/2，未知为 None。"""
    first = text.split(". ")[0]
    m = re.search(r"((?:upper|top|lower|bottom|middle|central|left|right|center)"
                  r"(?:[\s-]+(?:upper|top|lower|bottom|middle|central|left|right|center))*)\s+"
                  r"(?:corner|section|area|part|region|portion|side)", first)
    if not m:
        if "middle of the video" in first or "center of the video" in first:
            return (1, 1)
        return None
    words = m.group(1).split()
    row = col = None
    for w in words:
        if w in ("upper", "top"):
            row = 0
        elif w in ("lower", "bottom"):
            row = 2
        elif w == "left":
            col = 0
        elif w == "right":
            col = 2
    if row is None:
        row = 1
    if col is None:
        col = 1
    return (row, col)


# ---------------------------------------------------------------- 评估（batch_evaluate.py 口径）
def to_minmax(lwh, xyz):
    """中心 xyz + 尺寸 lwh → 轴对齐框 (xmin, ymin, zmin, xmax, ymax, zmax)。"""
    lwh = np.asarray(lwh, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.concatenate([xyz - lwh / 2, xyz + lwh / 2], axis=-1)


def iou_aabb(a, b):
    """两组轴对齐 3D 框（to_minmax 格式，按最后一维对齐）逐个的 IoU。"""
    inter = np.clip(np.minimum(a[..., 3:], b[..., 3:]) - np.maximum(a[..., :3], b[..., :3]), 0, None)
    vi = inter.prod(-1)
    va = (a[..., 3:] - a[..., :3]).prod(-1)
    vb = (b[..., 3:] - b[..., :3]).prod(-1)
    return vi / np.maximum(va + vb - vi, 1e-12)


def evaluate(pred, gt):
    """正常 3D AABB IoU 口径（中心 xyz ± lwh/2；由 v9→v11c 平台 ACE 变化推断平台用的就是这个）。
    pred/gt: {vid: (30,6) 数组，列为 l w h x y z}。按片段平均再对片段平均。
    PR10cm / PR5cm 为中心误差 <0.1 m / <0.05 m 的帧比例（平台的 PR@1.0 / PR@0.5 列对应这两项）。"""
    rows = []
    for vid, g in gt.items():
        p = pred[vid]
        bp, bg = to_minmax(p[:, :3], p[:, 3:]), to_minmax(g[:, :3], g[:, 3:])
        iou = iou_aabb(bp, bg)
        ce = np.linalg.norm(p[:, 3:] - g[:, 3:], axis=1)
        rows.append([100 * (iou > 0.5).mean(), 100 * (iou > 0.7).mean(), 100 * iou.mean(),
                     100 * (ce < 1.0).mean(), ce.mean(), 100 * (ce < 0.5).mean(),
                     100 * (ce < 0.1).mean(), 100 * (ce < 0.05).mean()])
    m = np.array(rows).mean(0)
    return dict(zip(["SR@0.5", "SR@0.7", "AOR", "PR@1.0", "ACE", "PR@0.5", "PR10cm", "PR5cm"],
                    np.round(m, 4).tolist()))


def gt_array(clip):
    """TRAIN 片段的 GT 数组 (30, 6)，每行 l w h x y z。"""
    return np.array([np.r_[f["lwh"], f["xyz"]] for f in clip["frames"]])


def evaluate_platform(pred, gt):
    """逐行复刻平台 batch_evaluate.py：把每行第 2-7 列 (l w h x y z) 的 (l,x)(w,y)(h,z) 两两排序后当作
    min/max 角点（预测与 GT 同样处理），再算 IoU、中心误差；PR10/PR05 的阈值是 0.1/0.05。
    平台页面上的 PR@1.0 / PR@0.5 对应这里的 PR10 / PR05。"""
    rows = []
    for vid, g in gt.items():
        p = pred[vid]
        lo_p, hi_p = np.minimum(p[:, :3], p[:, 3:]), np.maximum(p[:, :3], p[:, 3:])
        lo_g, hi_g = np.minimum(g[:, :3], g[:, 3:]), np.maximum(g[:, :3], g[:, 3:])
        inter = np.clip(np.minimum(hi_p, hi_g) - np.maximum(lo_p, lo_g), 0, None).prod(1)
        vp, vg = (hi_p - lo_p).prod(1), (hi_g - lo_g).prod(1)
        iou = np.where(vp + vg - inter > 0, inter / np.maximum(vp + vg - inter, 1e-12), 0.0)
        ce = np.linalg.norm((lo_p + hi_p) / 2 - (lo_g + hi_g) / 2, axis=1)
        rows.append([100 * (iou > 0.5).mean(), 100 * (iou > 0.7).mean(), 100 * iou.mean(),
                     100 * (ce < 0.1).mean(), ce.mean(), 100 * (ce < 0.05).mean()])
    m = np.array(rows).mean(0)
    return dict(zip(["SR@0.5", "SR@0.7", "AOR", "PR@1.0", "ACE", "PR@0.5"], np.round(m, 4).tolist()))
