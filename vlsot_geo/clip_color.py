"""用 CLIP（零样本）为每个检测框计算颜色概率，结果缓存到 cache/clip_color.pkl。
只用公开图像和检测框，不读取任何标签。
"""
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from geo_common import DATA, WORK

CLIP_DIR = os.environ.get("CLIP_DIR", WORK + "/weights/clip-vit-base-patch16")
COLOR_NAMES = ["white", "black", "silver grey", "red", "blue", "yellow-orange", "green", "brown"]
PROMPTS = {
    "white": ["a photo of a white vehicle", "a white car"],
    "black": ["a photo of a black vehicle", "a black car"],
    "silver grey": ["a photo of a silver gray vehicle", "a silver car", "a gray car"],
    "red": ["a photo of a red vehicle", "a red car"],
    "blue": ["a photo of a blue vehicle", "a blue car"],
    "yellow-orange": ["a photo of a yellow vehicle", "an orange car", "a yellow taxi"],
    "green": ["a photo of a green vehicle", "a green car"],
    "brown": ["a photo of a brown vehicle", "a brown car"],
}


def main():
    dets = json.load(open(WORK + "/cache/dets_yolo11x_1920.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_DIR).to(device).eval().half()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    with torch.no_grad():
        feats = []
        for name in COLOR_NAMES:
            t = proc(text=PROMPTS[name], return_tensors="pt", padding=True).to(device)
            tf = model.get_text_features(**t).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            feats.append(tf.mean(0))
        text = torch.stack(feats)
        text = text / text.norm(dim=-1, keepdim=True)
    out = {}
    keys = sorted(dets)
    batch_crops, batch_index = [], []

    def flush():
        if not batch_crops:
            return
        with torch.no_grad():
            px = proc(images=batch_crops, return_tensors="pt")["pixel_values"].to(device).half()
            f = model.get_image_features(pixel_values=px).float()
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ text.T).softmax(-1).cpu().numpy()
        for (k, j), row in zip(batch_index, p):
            out[k][j] = row.astype(np.float32)
        batch_crops.clear()
        batch_index.clear()

    for n, k in enumerate(keys):
        boxes = dets[k]
        out[k] = np.full((len(boxes), len(COLOR_NAMES)), 1.0 / len(COLOR_NAMES), dtype=np.float32)
        if not boxes:
            continue
        split, img_name = k.split("/")
        img = cv2.imread(f"{DATA}/{split}/img/{img_name}")
        h, w = img.shape[:2]
        for j, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)][:, :, ::-1]
            batch_crops.append(np.ascontiguousarray(crop))
            batch_index.append((k, j))
            if len(batch_crops) >= 512:
                flush()
        if n % 1000 == 0:
            print(n, len(keys), flush=True)
    flush()
    pickle.dump({"names": COLOR_NAMES, "probs": out}, open(WORK + "/cache/clip_color.pkl", "wb"))
    print("saved", len(out))


if __name__ == "__main__":
    main()
