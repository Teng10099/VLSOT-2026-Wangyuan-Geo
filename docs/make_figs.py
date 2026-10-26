"""生成技术报告 / PPT 用的配图（本地运行，只用我们自己的预测可视化与公开排行榜页面）。
输出到 docs/figs/：
  fig_leaderboard.png  排行榜（本队红框）
  fig_pipeline.png     方法流程图
  case_*.png           测试样例裁剪（绿=v12f boxfit 预测，青=v11c 射线法）
  fig_cases.png        样例拼图
用法：python make_figs.py <v12f viz 目录> <排行榜截图 png>"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
VIZ, LB = sys.argv[1], sys.argv[2]
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    """微软雅黑字体对象；bold=True 且粗体字体文件存在时用粗体。"""
    return ImageFont.truetype(FONT_B if bold and os.path.exists(FONT_B) else FONT, size)


# ---------------------------------------------------------------- 排行榜：裁到表格区域
lb = Image.open(LB).convert("RGB")
px = lb.load()
last = 0
for y in range(lb.height):
    row = [px[x, y] for x in range(0, lb.width, 7)]
    if any(sum(c) < 600 for c in row):
        last = y
lb.crop((0, 0, lb.width, min(lb.height, last + 24))).save(os.path.join(OUT, "fig_leaderboard.png"))

# ---------------------------------------------------------------- 测试样例裁剪
# v4_*.jpg：2500x900，2x2 个 1250x450 的格子（帧 0/10/20/29），每格左 800 为全图、右 450 为目标放大
TILE = {0: (0, 0), 10: (1250, 0), 20: (0, 450), 29: (1250, 450)}
# 均为第 0 帧（033447 另取第 29 帧）；括号内为描述文本给出的该帧距离（测试输入，不是标签）
CASES = [("003235", 0, "003235 截断货车（16.5 m）"), ("001483", 0, "001483 角落截断轿车（14.7 m）"),
         ("000237", 0, "000237 面包车（22.2 m）"), ("006543", 0, "006543 扁框轿车 h=0.6（21.7 m）"),
         ("033447", 0, "033447 横穿运动 第 0 帧（42.0 m）"), ("033447", 29, "033447 横穿运动 第 29 帧（46.8 m）")]
crops = []
for vid, fi, title in CASES:
    im = Image.open(os.path.join(VIZ, f"v4_{vid}.jpg")).convert("RGB")
    x0, y0 = TILE[fi]
    crop = im.crop((x0 + 800, y0, x0 + 1250, y0 + 450))
    full = im.crop((x0, y0, x0 + 800, y0 + 450))
    crop.save(os.path.join(OUT, f"case_{vid}_f{fi}_zoom.png"))
    full.save(os.path.join(OUT, f"case_{vid}_f{fi}_full.png"))
    crops.append((crop, title))

W, H, CAP = 450, 450, 58
grid = Image.new("RGB", (3 * W + 4 * 12, 2 * (H + CAP) + 3 * 12), "white")
d = ImageDraw.Draw(grid)
for k, (crop, title) in enumerate(crops):
    r, c = divmod(k, 3)
    x, y = 12 + c * (W + 12), 12 + r * (H + CAP + 12)
    grid.paste(crop, (x, y))
    d.text((x + 4, y + H + 12), title, fill=(16, 42, 67), font=font(24))
grid.save(os.path.join(OUT, "fig_cases.png"))

# ---------------------------------------------------------------- 流程图
BOXES = [
    ("输入", ["30 帧单目图像 1920×1080", "英文描述 + 颜色/状态/类别字段", "相机标定（V2X-Seq-SPD）"]),
    ("① 文本解析", ["尺寸 l/w/h；第 0/14/29 帧距离", "航向与朝向词、九宫格区域、名次", "截断、最近车及其相对方位"]),
    ("② 二维检测", ["YOLO11x（COCO 预训练）", "置信度 ≥0.05 的车辆框都作候选", "EFN 颜色分类器给出颜色概率"]),
    ("③ 候选打分", ["9 个特征：距离一致、区域、", "名次、颜色、类别、截断、方位…", "线性排序模型（TRAIN 拟合）"]),
    ("④ 全局跟踪", ["Viterbi 动态规划选整条轨迹", "相邻帧框连续", "运动方向与文本航向一致"]),
    ("⑤ boxfit 三维定位", ["文本尺寸 + 航向 + 距离", "3D 框投影对齐检测框未截断边", "地平面作为高度弱先验"]),
    ("⑥ 时序平滑", ["停车：30 帧取中位数", "运动：保持文本距离曲线，", "方位角做局部线性平滑"]),
    ("输出", ["每帧 l w h x y z", "virtuallidar 坐标系", "result/<视频ID>.txt"]),
]
BW, BH, GAP = 520, 250, 70
fig = Image.new("RGB", (4 * BW + 3 * GAP + 80, 2 * BH + 150 + 40), "white")
d = ImageDraw.Draw(fig)
pos = []
for k in range(8):
    r, c = divmod(k, 4)
    x, y = 40 + c * (BW + GAP), 40 + r * (BH + 150)
    pos.append((x, y))
    hl = k in (3, 5)
    fill = (232, 244, 241) if hl else ((233, 241, 247) if k in (0, 7) else (247, 250, 252))
    edge = (15, 118, 110) if hl else (19, 101, 166)
    d.rounded_rectangle((x, y, x + BW, y + BH), radius=22, fill=fill, outline=edge, width=4)
    title, lines = BOXES[k]
    d.text((x + 24, y + 18), title, fill=edge, font=font(36, True))
    for j, line in enumerate(lines):
        d.text((x + 24, y + 82 + j * 50), line, fill=(16, 42, 67), font=font(29))


def arrow(x1, y1, x2, y2, color=(82, 104, 125)):
    """在流程图上画一条带箭头的线段（终点处画两条 26 像素的箭翼）。"""
    d.line((x1, y1, x2, y2), fill=color, width=6)
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    for s in (+0.45, -0.45):
        d.line((x2, y2, x2 - 26 * math.cos(a + s), y2 - 26 * math.sin(a + s)), fill=color, width=6)


for k in (0, 1, 2, 4, 5, 6):
    x, y = pos[k]
    arrow(x + BW + 6, y + BH / 2, x + BW + GAP - 8, y + BH / 2)
# 第一行末尾 → 第二行开头（折线）
x3, y3 = pos[3]
x4, y4 = pos[4]
mid = y3 + BH + 75
d.line((x3 + BW / 2, y3 + BH + 4, x3 + BW / 2, mid), fill=(82, 104, 125), width=6)
d.line((x3 + BW / 2, mid, x4 + BW / 2, mid), fill=(82, 104, 125), width=6)
arrow(x4 + BW / 2, mid, x4 + BW / 2, y4 - 8)
fig.save(os.path.join(OUT, "fig_pipeline.png"))

# ---------------------------------------------------------------- 流程图（竖版，给 A4 纵向的 docx 用：4 行 × 2 列，蛇形顺序）
VGAP_X, VGAP_Y = 90, 110
GRID = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (3, 0)]  # 第 k 个框的 (行, 列)
figv = Image.new("RGB", (2 * BW + VGAP_X + 80, 4 * BH + 3 * VGAP_Y + 80), "white")
d = ImageDraw.Draw(figv)
posv = []
for k, (r, c) in enumerate(GRID):
    x, y = 40 + c * (BW + VGAP_X), 40 + r * (BH + VGAP_Y)
    posv.append((x, y))
    hl = k in (3, 5)
    fill = (232, 244, 241) if hl else ((233, 241, 247) if k in (0, 7) else (247, 250, 252))
    edge = (15, 118, 110) if hl else (19, 101, 166)
    d.rounded_rectangle((x, y, x + BW, y + BH), radius=22, fill=fill, outline=edge, width=4)
    title, lines = BOXES[k]
    d.text((x + 24, y + 18), title, fill=edge, font=font(36, True))
    for j, line in enumerate(lines):
        d.text((x + 24, y + 82 + j * 50), line, fill=(16, 42, 67), font=font(29))
for k in range(7):
    (x1, y1), (x2, y2) = posv[k], posv[k + 1]
    if y1 == y2:  # 同一行：左右箭头
        if x2 > x1:
            arrow(x1 + BW + 6, y1 + BH / 2, x2 - 8, y2 + BH / 2)
        else:
            arrow(x1 - 6, y1 + BH / 2, x2 + BW + 8, y2 + BH / 2)
    else:  # 换行：同一列向下
        arrow(x1 + BW / 2, y1 + BH + 6, x2 + BW / 2, y2 - 8)
figv.save(os.path.join(OUT, "fig_pipeline_v.png"))
print("figures written to", OUT, sorted(os.listdir(OUT)))
