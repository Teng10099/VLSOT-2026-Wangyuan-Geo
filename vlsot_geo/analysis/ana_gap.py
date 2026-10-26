"""交叉验证（TRAIN 5 折，SR@0.5 88.41）与平台测试集（61.67）差距的分析。只用 TRAIN 标签、测试集公开描述和我们自己的预测，
不涉及任何测试标签。技术报告“误差上限与局限”一节与 PPT 第 5 页的数字来自本脚本。

  1. 选对 / 没选对目标的片段各自的 SR@0.5 与 ACE。
  2. 从 227 段中随机抽 10 段，平均 SR@0.5 不高于平台成绩的概率；按测试集的情形构成加权的预期。
  3. 朝向：提交格式没有朝向，我们的框长边沿 x 轴。若平台按真值的实际朝向计算 IoU，求各测试段在
     “中心与尺寸完全正确”时的 IoU 上限；以及同一口径下 TRAIN 交叉验证的 SR@0.5。
  4. 误差移植：每个测试段从同类（情形 + 停/动）TRAIN 段中随机取一段的逐帧误差（预测 - GT）和尺寸比例，
     构造模拟真值，分别按 A 轴对齐、B 真值按文本航向旋转 两种口径算 6 项指标，与平台实际值比较。

用法（在 vlsot_geo 目录下）：
  PYTHONPATH=. python analysis/ana_gap.py [--dumps "$VLSOT_WORK/runs/ablation/a9_v12f_f?.json"] [--test_out $VLSOT_WORK/out/test_v12f]
--dumps 为 run_ablation.sh 生成的 v12f 交叉验证 dump；--test_out 为 run_final.sh 的测试输出目录（含 tracks.json 与 result.zip 或 result/）。
"""
import argparse
import glob
import io
import json
import math
import os
import zipfile

import numpy as np

from geo_common import WORK, load_clips, parse_text

# 平台方法页 301 上 v12f（2026-10-24 16:30）的成绩
PLATFORM = {"SR@0.5": 61.6667, "SR@0.7": 38.3333, "AOR": 57.8363, "ACE": 0.2259, "PR@1.0": 16.6667, "PR@0.5": 1.6667}
KEYS = list(PLATFORM)
KINDS = ["近处截断", "薄框", "远距离", "其他截断", "普通"]

parser = argparse.ArgumentParser()
parser.add_argument("--dumps", default=WORK + "/runs/ablation/a9_v12f_f?.json")
parser.add_argument("--test_out", default=WORK + "/out/test_v12f")
parser.add_argument("--draws", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def rect(cx, cy, l, w, yaw):
    """底面矩形的 4 个顶点（逆时针），l 沿航向 yaw。"""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in ((l / 2, w / 2), (-l / 2, w / 2), (-l / 2, -w / 2), (l / 2, -w / 2))]


def clip_poly(subject, clipper):
    """Sutherland–Hodgman 算法求两个凸多边形的交（顶点均为逆时针）。"""
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            sq = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def area(poly):
    """多边形面积（鞋带公式）。"""
    n = len(poly)
    return 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) if n >= 3 else 0.0


def iou3d(p, g, yaw_g):
    """3D IoU：p、g 为 l w h x y z；p 按朝向 0 放置（提交格式没有朝向），g 按 yaw_g 旋转，高度方向轴对齐。
    yaw_g = 0 时就是交叉验证用的轴对齐 IoU。"""
    bev = area(clip_poly(rect(p[3], p[4], p[0], p[1], 0.0), rect(g[3], g[4], g[0], g[1], yaw_g)))
    hz = max(0.0, min(p[5] + p[2] / 2, g[5] + g[2] / 2) - max(p[5] - p[2] / 2, g[5] - g[2] / 2))
    inter = bev * hz
    return inter / (p[0] * p[1] * p[2] + g[0] * g[1] * g[2] - inter)


def metrics(ious, ces):
    """一段的 6 项指标（PR 阈值 0.1 m / 0.05 m，与平台对本方法给出的数值同口径）。"""
    ious, ces = np.asarray(ious), np.asarray(ces)
    return {"SR@0.5": 100 * (ious > 0.5).mean(), "SR@0.7": 100 * (ious > 0.7).mean(), "AOR": 100 * ious.mean(),
            "ACE": ces.mean(), "PR@1.0": 100 * (ces < 0.1).mean(), "PR@0.5": 100 * (ces < 0.05).mean()}


def edge_frac(boxes, m=10):
    """选中框贴图像边界（m 像素内）的帧比例；只用我们自己的选择结果，TRAIN 与 TEST 同一口径。"""
    return float(np.mean([b is not None and (b[0] < m or b[1] < m or b[2] > 1920 - m or b[3] > 1080 - m) for b in boxes]))


def kind(desc, sel):
    """按测试时也能得到的信息（文本尺寸与距离、选中框是否贴边）把片段归为一类，优先级见 KINDS 的顺序。"""
    t = parse_text(desc)
    lwh = t["lwh"] if t["lwh"] is not None else np.array([4.5, 1.8, 1.5])
    r0 = t["distances"][0] if t["distances"] else 30.0
    e = edge_frac(sel)
    if r0 < 20 and e >= 0.3:
        return "近处截断"
    if min(lwh[1], lwh[2]) < 1.0:
        return "薄框"
    if r0 >= 80:
        return "远距离"
    if e >= 0.3:
        return "其他截断"
    return "普通"


def yaw_of(desc):
    """文本给出的航向（azimuth，virtuallidar 下 0 = +x），没有时为 0。"""
    az = parse_text(desc)["azimuth"]
    return math.radians(az) if az is not None else 0.0


def fmt(k, v):
    """ACE 保留 3 位小数，其余 1 位。"""
    return f"{v:.3f}" if k == "ACE" else f"{v:.1f}"


def main():
    """读取交叉验证 dump 与测试输出，依次打印 4 部分分析。"""
    rng = np.random.RandomState(args.seed)
    cv = {}
    for path in sorted(glob.glob(args.dumps)):
        cv.update(json.load(open(path)))
    rows = []
    for v, x in cv.items():
        pr, gt = np.array(x["pred"]), np.array(x["gt"])
        ce = np.linalg.norm(pr[:, 3:] - gt[:, 3:], axis=1)
        yaw = yaw_of(x["desc"])
        rows.append({"m": metrics([iou3d(pr[f], gt[f], 0.0) for f in range(30)], ce),
                     "m_rot": metrics([iou3d(pr[f], gt[f], yaw) for f in range(30)], ce),
                     "sel": np.mean([s is not None and b is not None and s == b for s, b in zip(x["sel"], x["gtbox"])]),
                     "kind": kind(x["desc"], x["sel"]), "mov": "mov" in (x["state"] or "").lower(),
                     "az": abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))),
                     "err": pr[:, 3:] - gt[:, 3:], "scale": gt[:, :3] / pr[:, :3]})
    n = len(rows)
    mean = lambda rs, k, key="m": float(np.mean([r[key][k] for r in rs]))  # noqa: E731
    print(f"1. TRAIN 交叉验证 {n} 段：" + "  ".join(f"{k} {fmt(k, mean(rows, k))}" for k in KEYS))
    for name, rs in (("选对目标（≥80% 帧）", [r for r in rows if r["sel"] >= 0.8]), ("没选对", [r for r in rows if r["sel"] < 0.8])):
        print(f"   {name}：{len(rs)} 段，SR@0.5 {mean(rs, 'SR@0.5'):.2f}，ACE {mean(rs, 'ACE'):.3f} m")

    # 测试集：我们提交的预测与选中框
    tracks = json.load(open(os.path.join(args.test_out, "tracks.json")))
    zpath = os.path.join(args.test_out, "result.zip")
    zf = zipfile.ZipFile(zpath) if os.path.exists(zpath) else None
    test = []
    for c in load_clips("test"):
        text = zf.read(f"result/{c['vid']}.txt").decode() if zf else open(os.path.join(args.test_out, "result", c["vid"] + ".txt")).read()
        pred = np.loadtxt(io.StringIO(text))[:, 1:]
        test.append({"vid": c["vid"], "pred": pred, "yaw": yaw_of(c["desc"]), "az": parse_text(c["desc"])["azimuth"],
                     "kind": kind(c["desc"], tracks[c["vid"]]), "mov": "mov" in (c["state"] or "").lower()})

    sr = np.array([r["m"]["SR@0.5"] for r in rows])
    rnd = np.array([sr[rng.choice(n, 10, replace=False)].mean() for _ in range(args.draws)])
    print(f"\n2. 随机抽 10 段：平均 SR@0.5 中位数 {np.median(rnd):.2f}，≤ {PLATFORM['SR@0.5']:.2f} 的概率 "
          f"{100 * np.mean(rnd <= PLATFORM['SR@0.5']):.2f}%")
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
    for k in KINDS:
        tv = [t["vid"] for t in test if t["kind"] == k]
        print(f"   {k:<6} TRAIN {len(by_kind[k]):3d} 段  CV SR@0.5 {mean(by_kind[k], 'SR@0.5') if by_kind[k] else float('nan'):6.2f}"
              f"   测试 {len(tv)} 段 {tv}")
    est = np.mean([mean(by_kind[t["kind"]], "SR@0.5") for t in test])
    print(f"   按测试集情形构成加权的预期 SR@0.5：{est:.2f}")

    print("\n3. 朝向（测试集文本航向；“中心与尺寸完全正确”时，真值按航向旋转后与提交框的 IoU 上限）")
    for t in test:
        cap = iou3d(t["pred"][0], t["pred"][0], t["yaw"])
        t["cap"] = cap
        print(f"   {t['vid']}  航向 {t['az']!s:>6}  {'动' if t['mov'] else '停'}  {t['kind']:<6}  IoU 上限 {cap:.2f}{'  < 0.5' if cap < 0.5 else ''}")
    az = np.array([r["az"] for r in rows])
    print(f"   TRAIN 中 |航向| 在 20°~160° 之间的片段：{100 * np.mean((az > 20) & (az < 160)):.0f}%；"
          f"测试集：{sum(1 for t in test if t['cap'] < 0.5)} / 10 段 IoU 上限 < 0.5")
    print(f"   若 TRAIN 真值也按文本航向旋转，交叉验证 SR@0.5 {mean(rows, 'SR@0.5', 'm_rot'):.2f}"
          f"（轴对齐 {mean(rows, 'SR@0.5'):.2f}）")
    n_ok = sum(1 for t in test if t["cap"] >= 0.5)
    print(f"   平台达标帧 {PLATFORM['SR@0.5'] * 3:.0f} / 300；若上限 < 0.5 的段全部不达标，其余 {n_ok} 段的达标率 "
          f"{100 * PLATFORM['SR@0.5'] * 3 / (30 * n_ok):.1f}%")

    # 误差移植：同类且停/动相同的 TRAIN 段不足 5 段时，退回只按类别
    pools = {}
    for t in test:
        same = [r for r in rows if r["kind"] == t["kind"] and r["mov"] == t["mov"]]
        pools[t["vid"]] = same if len(same) >= 5 else by_kind[t["kind"]]
    sim = {"A": {}, "B": {}}
    for t in test:
        p = t["pred"]
        for name in sim:
            sim[name][t["vid"]] = []
        for r in pools[t["vid"]]:
            g = p.copy()
            g[:, 3:] = p[:, 3:] - r["err"]
            g[:, :3] = p[:, :3] * r["scale"]
            ce = np.linalg.norm(r["err"], axis=1)
            sim["A"][t["vid"]].append(metrics([iou3d(p[f], g[f], 0.0) for f in range(30)], ce))
            sim["B"][t["vid"]].append(metrics([iou3d(p[f], g[f], t["yaw"]) for f in range(30)], ce))
    print(f"\n4. 误差移植（{args.draws} 次，中位数 [5%, 95%]）")
    print("   " + " " * 16 + "".join(f"{k:>22}" for k in KEYS))
    print("   平台实际        " + "".join(f"{fmt(k, PLATFORM[k]):>22}" for k in KEYS))
    for name, label in (("A", "A 轴对齐"), ("B", "B 真值按航向旋转")):
        draws = [{k: np.mean([sim[name][t["vid"]][rng.randint(len(sim[name][t["vid"]]))][k] for t in test]) for k in KEYS}
                 for _ in range(args.draws)]
        cells = []
        for k in KEYS:
            a = np.array([d[k] for d in draws])
            cells.append(f"{fmt(k, np.median(a))} [{fmt(k, np.percentile(a, 5))}, {fmt(k, np.percentile(a, 95))}]")
        print(f"   {label:<14}" + "".join(f"{c:>22}" for c in cells))
    ok7 = [t for t in test if t["cap"] >= 0.5]
    d7 = [np.mean([sim["A"][t["vid"]][rng.randint(len(sim["A"][t["vid"]]))]["SR@0.5"] for t in ok7]) for _ in range(args.draws)]
    print(f"   上限 ≥ 0.5 的 {len(ok7)} 段在 A 口径下的预期 SR@0.5：{np.median(d7):.1f} [{np.percentile(d7, 5):.1f}, {np.percentile(d7, 95):.1f}]")


if __name__ == "__main__":
    main()
