"""停车目标 GT 的逐帧抖动，以及"距离用哪种方式取"对中心误差的影响（只用 TRAIN 标签）。
A. GT 中心相对片段中位数的偏差（|dx| |dy| |dz| 与 3D）：常数预测能达到的误差下限、<0.1m / <0.05m 的比例上限。
B. 径向距离：常数（插值曲线中位数 / 三个文本距离中位数）vs 逐帧插值（线性 / 二次），与 GT 水平距离逐帧对比。
C. GT 尺寸逐帧变化：相对文本尺寸的偏差；以及"中心完美、尺寸用文本"时的 IoU 上限。"""
import numpy as np

from geo_common import load_clips, parse_text

clips = load_clips("train")


def iou(lp, cp, lg, cg):
    lo = np.maximum(cp - lp / 2, cg - lg / 2)
    hi = np.minimum(cp + lp / 2, cg + lg / 2)
    inter = np.clip(hi - lo, 0, None).prod(-1)
    return inter / (lp.prod(-1) + lg.prod(-1) - inter)


for state in ("park", "mov"):
    dev, rad, ious, dims = [], {"med_curve": [], "med3": [], "lin": [], "quad": []}, [], []
    n = 0
    for c in clips:
        if (state == "park") != ("park" in (c["state"] or "").lower()):
            continue
        t = parse_text(c["desc"])
        if t["lwh"] is None or len(t["distances"]) < 3:
            continue
        G = np.array([f["xyz"] for f in c["frames"]])
        L = np.array([f["lwh"] for f in c["frames"]])
        d = np.array(t["distances"][:3])
        rg = np.hypot(G[:, 0], G[:, 1])
        if np.abs(d - rg[[0, 14, 29]]).max() > 0.3:
            continue
        n += 1
        lin = np.interp(np.arange(30), [0, 14, 29], d)
        quad = np.polyval(np.polyfit([0, 14, 29], d, 2), np.arange(30))
        rad["lin"].append(lin - rg)
        rad["quad"].append(quad - rg)
        rad["med_curve"].append(np.full(30, np.median(lin)) - rg)
        rad["med3"].append(np.full(30, np.median(d)) - rg)
        if state == "park":
            dev.append(G - np.median(G, 0))
        ious.append(iou(np.repeat(t["lwh"][None], 30, 0), G, L, G))
        dims.append(L - t["lwh"])
    print(f"===== {state}: clips {n}")
    if dev:
        D = np.concatenate(dev)
        e = np.linalg.norm(D, axis=1)
        print("A. GT jitter around clip median: mean |dx| |dy| |dz|", np.abs(D).mean(0).round(3),
              " 3D mean %.3f  frac<0.1 %.2f  frac<0.05 %.2f" % (e.mean(), (e < 0.1).mean(), (e < 0.05).mean()))
    print("B. radial error (pred range - GT range):")
    for k, v in rad.items():
        v = np.abs(np.concatenate(v))
        print(f"   {k:10s} mean {v.mean():.3f}  median {np.median(v):.3f}  frac<0.05 {np.mean(v < 0.05):.2f}  frac<0.1 {np.mean(v < 0.1):.2f}")
    I = np.concatenate(ious)
    Dm = np.abs(np.concatenate(dims))
    print("C. |GT dims - text dims| mean l w h", Dm.mean(0).round(3), " IoU with perfect centre: mean %.3f  frac<0.5 %.3f  frac<0.7 %.3f"
          % (I.mean(), (I < 0.5).mean(), (I < 0.7).mean()))
