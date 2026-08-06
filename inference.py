from ultralytics import YOLO
from PIL import Image
from inference_helpers import run_tiled_inference, rotated_nms
from inference_config import MODEL_PATH, CONF, TILE_SIZE, PIXEL_THRESHOLD
import os


def run_inference(model, source_path, conf=CONF, tile_size=TILE_SIZE,
                   pixel_threshold=PIXEL_THRESHOLD, base_out_dir=None):
    """
    Runs inference on source_path, auto-choosing tiled vs normal based on pixel_threshold.
    Returns (detections, mode) where detections is a list of {"poly", "conf", "cls"}
    in global image coordinates, and mode is "tiled" or "normal".
    If base_out_dir is given, also saves a visual result image to base_out_dir/<mode>/resultN.png.
    """
    with Image.open(source_path) as img:
        width, height = img.size
    pixel_count = width * height
    print(f"Image size: {width}x{height} ({pixel_count:,} pixels)")

    if pixel_count > pixel_threshold:
        print("Image exceeds threshold: running tiled inference...")
        mode = "tiled"
        all_detections, _ = run_tiled_inference(model, source_path, tile_size=tile_size, conf=conf)
        print(f"Raw detections before NMS: {len(all_detections)}")
        detections = rotated_nms(all_detections, iou_threshold=0.4)
        print(f"Final detections after NMS: {len(detections)}")
    else:
        print("Image below threshold: running normal inference...")
        mode = "normal"
        results = model.predict(source=source_path, conf=conf, save=False, verbose=False)
        r = results[0]
        detections = []
        if r.obb is not None and len(r.obb) > 0:
            polys = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            clss = r.obb.cls.cpu().numpy()
            for poly, c, cl in zip(polys, confs, clss):
                detections.append({"poly": poly, "conf": float(c), "cls": int(cl)})

    if base_out_dir:
        from inference_helpers import draw_detections
        out_dir = os.path.join(base_out_dir, mode)
        os.makedirs(out_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(out_dir, f"result{i}.png")):
            i += 1
        out_path = os.path.join(out_dir, f"result{i}.png")
        draw_detections(source_path, detections, model.names, out_path)
        print(f"Saved to: {out_path}")

    return detections, mode


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    run_inference(
        model,
        source_path="model-input/orthomosaic.png",
        base_out_dir="model-output",
    )