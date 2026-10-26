"""核对 ONNX 推理路径与原始权重是否一致（只比较我们自己的检测框、颜色概率与预测结果，不涉及任何标签）。

  python check_onnx.py dets A.json B.json [--prefix test/]
      比较两份检测缓存：每张图按类别一致、IoU 最大的原则一一匹配，统计匹配率、IoU、坐标与置信度差。
  python check_onnx.py efn A.pkl B.pkl --dets_a A.json [--dets_b B.json]
      比较两份 EFN 颜色概率；两份概率基于不同检测框时先按 IoU 匹配框。
  python check_onnx.py results out/test_A out/test_B
      比较两次 test 输出：每个 result/<vid>.txt 是否逐字节相同、l w h x y z 的最大差、选中框的 IoU。
--json 路径：把统计写成 json。
"""
import argparse
import hashlib
import json
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("what", choices=["dets", "efn", "results"])
parser.add_argument("a")
parser.add_argument("b")
parser.add_argument("--prefix", default="", help="只比较以此开头的图像键，如 test/")
parser.add_argument("--dets_a", default="")
parser.add_argument("--dets_b", default="")
parser.add_argument("--json", default="")
args = parser.parse_args()


def iou_matrix(a, b):
    """两组 2D 框 (x1, y1, x2, y2, ...) 两两之间的 IoU 矩阵，形状 (len(a), len(b))。"""
    a, b = np.asarray(a, dtype=np.float64)[:, :4], np.asarray(b, dtype=np.float64)[:, :4]
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def match(a, b, min_iou=0.5):
    """贪心一一匹配（同类别），返回 [(i, j, iou)]。"""
    if not len(a) or not len(b):
        return []
    m = iou_matrix(a, b)
    same = np.array([[int(x[5]) == int(y[5]) for y in b] for x in a])
    m = np.where(same, m, 0.0)
    pairs = []
    while True:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        if m[i, j] < min_iou:
            break
        pairs.append((int(i), int(j), float(m[i, j])))
        m[i, :] = 0.0
        m[:, j] = 0.0
    return pairs


def pct(v, q):
    """v 的第 q 百分位数；v 为空时返回 NaN。"""
    return float(np.percentile(v, q)) if len(v) else float("nan")


def cmp_dets(A, B):
    """比较两份检测缓存（图像键 → 框列表）：同类别框按 IoU ≥ 0.5 贪心一一匹配，
    统计逐图完全相同的数量、匹配数、未匹配的高置信度（≥ 0.25）框、匹配框的 IoU、坐标差与置信度差。"""
    keys = sorted(k for k in set(A) & set(B) if k.startswith(args.prefix))
    ious, dxy, dconf, n_a, n_b, n_m, un_a25, un_b25, exact = [], [], [], 0, 0, 0, 0, 0, 0
    for k in keys:
        a, b = A[k], B[k]
        n_a += len(a)
        n_b += len(b)
        pairs = match(a, b)
        n_m += len(pairs)
        ia = {i for i, _, _ in pairs}
        ib = {j for _, j, _ in pairs}
        un_a25 += sum(1 for i, x in enumerate(a) if i not in ia and x[4] >= 0.25)
        un_b25 += sum(1 for j, y in enumerate(b) if j not in ib and y[4] >= 0.25)
        exact += int(a == b)
        for i, j, v in pairs:
            ious.append(v)
            dxy.append(np.abs(np.array(a[i][:4]) - np.array(b[j][:4])).max())
            dconf.append(abs(a[i][4] - b[j][4]))
    s = {"images": len(keys), "images_identical": exact, "boxes_a": n_a, "boxes_b": n_b, "matched": n_m,
         "unmatched_conf>=0.25_a": un_a25, "unmatched_conf>=0.25_b": un_b25,
         "iou_mean": float(np.mean(ious)) if ious else None, "iou_p1": pct(ious, 1), "iou_p5": pct(ious, 5),
         "max_abs_coord_px_median": pct(dxy, 50), "max_abs_coord_px_p99": pct(dxy, 99),
         "abs_conf_median": pct(dconf, 50), "abs_conf_p99": pct(dconf, 99)}
    return s


def cmp_efn(PA, PB, DA, DB):
    """比较两份 EFN 颜色概率：DB 为 None 时按框序号一一对应，否则先按 IoU ≥ 0.9 匹配两份检测框；
    跳过含 NaN 的框，统计最可能颜色一致的个数与概率的最大绝对差。"""
    keys = sorted(k for k in set(PA) & set(PB) if k.startswith(args.prefix))
    diffs, agree, n = [], 0, 0
    for k in keys:
        pa, pb = PA[k], PB[k]
        if DB is None:
            pairs = [(i, i, 1.0) for i in range(min(len(pa), len(pb)))]
        else:
            pairs = match(DA[k], DB[k], min_iou=0.9)
        for i, j, _ in pairs:
            x, y = pa[i], pb[j]
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            n += 1
            diffs.append(float(np.abs(x - y).max()))
            agree += int(np.argmax(x) == np.argmax(y))
    return {"images": len(keys), "boxes_compared": n, "argmax_agree": agree,
            "max_abs_prob_median": pct(diffs, 50), "max_abs_prob_p99": pct(diffs, 99),
            "max_abs_prob_max": max(diffs) if diffs else None}


def cmp_results(a, b):
    """比较两次 test 输出目录：每个 result/<vid>.txt 是否逐字节相同、l w h 与 x y z 的最大差、
    中心平均距离，以及两次逐帧选中框的最小 IoU。"""
    ta = json.load(open(f"{a}/tracks.json"))
    tb = json.load(open(f"{b}/tracks.json"))
    rows, same_bytes = [], 0
    for vid in sorted(ta):
        fa, fb = f"{a}/result/{vid}.txt", f"{b}/result/{vid}.txt"
        ha = hashlib.sha256(open(fa, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(fb, "rb").read()).hexdigest()
        same_bytes += int(ha == hb)
        pa, pb = np.loadtxt(fa), np.loadtxt(fb)
        d = np.abs(pa[:, 1:] - pb[:, 1:])
        ious = [iou_matrix([x], [y])[0, 0] if x is not None and y is not None else 0.0 for x, y in zip(ta[vid], tb[vid])]
        rows.append({"vid": vid, "identical": ha == hb, "max_abs_lwh": float(d[:, :3].max()),
                     "max_abs_xyz": float(d[:, 3:].max()), "mean_center_dist": float(np.linalg.norm(pa[:, 4:] - pb[:, 4:], axis=1).mean()),
                     "min_sel_iou": float(min(ious))})
    return {"files": len(rows), "identical_files": same_bytes, "per_clip": rows}


if args.what == "dets":
    out = cmp_dets(json.load(open(args.a)), json.load(open(args.b)))
elif args.what == "efn":
    da = json.load(open(args.dets_a)) if args.dets_a else None
    db = json.load(open(args.dets_b)) if args.dets_b else None
    out = cmp_efn(pickle.load(open(args.a, "rb"))["probs"], pickle.load(open(args.b, "rb"))["probs"], da, db)
else:
    out = cmp_results(args.a, args.b)
print(json.dumps(out, ensure_ascii=False, indent=1))
if args.json:
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
