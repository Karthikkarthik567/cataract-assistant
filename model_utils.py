import os
import csv
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ============================
# LOAD MODEL (LAZY IMPORT)
# ============================
def load_model(model_path="model/best_model.pt", class_names=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Lazy import (prevents crash at startup)
    from ultralytics import YOLO

    model = YOLO(model_path)
    names = class_names or {0: "cataract", 1: "normal"}

    return {"model": model, "names": names}


# ============================
# INFERENCE (SAFE MODE)
# ============================
def run_inference(model_bundle, image_path):
    model = model_bundle["model"]
    names = model_bundle["names"]

    # Use predict API (more stable)
    results = model.predict(source=image_path, save=False)

    r = results[0]
    boxes_out = []

    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            boxes_out.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class": cls
            })

    if boxes_out:
        top = max(boxes_out, key=lambda b: b["confidence"])
        label = names.get(top["class"], "unknown")
        confidence = top["confidence"]
    else:
        label = "no-detection"
        confidence = 0.0

    # Logging
    os.makedirs("logs", exist_ok=True)
    with open("logs/predictions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), image_path, label, confidence])

    return {
        "label": label,
        "confidence": confidence,
        "boxes": boxes_out,
        "names": names
    }


# ============================
# DRAW BOXES (NO OPENCV)
# ============================
def draw_boxes_on_image(src, boxes, dst, labels=None):
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cls = b["class"]
        conf = b["confidence"]

        label = f"{labels.get(cls, cls)}: {conf:.2f}" if labels else f"{cls}: {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red", font=font)

    img.save(dst)
    return dst
