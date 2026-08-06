from ultralytics import YOLO
from PIL import Image
from inference_helpers import run_tiled_inference, rotated_nms, draw_detections
import os

model = YOLO("model/yolov8-obb.pt")

source_path = "model-input/largelot.png"
conf = 0.50
tile_size = 700
pixel_threshold = 4_000_000
base_out_dir = "model-output"


def get_output_path(base_dir, subfolder, filename_prefix, ext="png"):
    out_dir = os.path.join(base_dir, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(out_dir, f"{filename_prefix}{i}.{ext}")):
        i += 1
    return os.path.join(out_dir, f"{filename_prefix}{i}.{ext}")


# Checking image size
with Image.open(source_path) as img:
    width, height = img.size
pixel_count = width * height
print(f"Image size: {width}x{height} ({pixel_count:,} pixels)")

class_names = model.names

# Running inference
if pixel_count > pixel_threshold:
    print("Image exceeds threshold: running tiled inference...")

    all_detections, (w, h) = run_tiled_inference(model, source_path, tile_size=tile_size, conf=conf)
    print(f"Raw detections before NMS: {len(all_detections)}")

    final_detections = rotated_nms(all_detections, iou_threshold=0.4)
    print(f"Final detections after NMS: {len(final_detections)}")

    out_path = get_output_path(base_out_dir, "tiled", "result")
    draw_detections(source_path, final_detections, class_names, out_path)
    print(f"Saved to: {out_path}")

else:
    print("Image below threshold: running normal inference...")

    results = model.predict(
        source=source_path,
        conf=conf,
        save=False,
    )

    plotted = results[0].plot()
    out_path = get_output_path(base_out_dir, "normal", "result")
    Image.fromarray(plotted[..., ::-1]).save(out_path)  # BGR -> RGB
    print(f"Saved to: {out_path}")