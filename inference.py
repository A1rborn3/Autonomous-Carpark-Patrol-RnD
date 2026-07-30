from ultralytics import YOLO

model = YOLO("model/yolov8-obb.pt")

results = model.predict(
    source="model-input/orthomosaic_clahe.png",
    conf=0.25,
    save=True,
    project="model-output",
    name="lot1"
)

print(f"Saved to: {results[0].save_dir}")