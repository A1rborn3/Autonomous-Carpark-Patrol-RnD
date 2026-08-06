import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

def generate_tiles(width, height, tile_size, offset_x=0, offset_y=0):
    """Generate tile coordinates covering the image, starting from an offset."""
    tiles = []
    y = -offset_y
    while y < height:
        x = -offset_x
        while x < width:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + tile_size, width), min(y + tile_size, height)
            if (x2 - x1) > 50 and (y2 - y1) > 50:  # skip tiny sliver tiles at edges
                tiles.append((x1, y1, x2, y2))
            x += tile_size
        y += tile_size
    return tiles


def run_tiled_inference(model, image_path, tile_size=640, conf=0.25):
    """Runs inference over two offset grids and returns all raw detections in global image coordinates."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    all_detections = []
    grid_offsets = [(0, 0), (tile_size // 2, tile_size // 2)]  # grid A, grid B (shifted)

    for offset_x, offset_y in grid_offsets:
        tiles = generate_tiles(width, height, tile_size, offset_x, offset_y)
        for (x1, y1, x2, y2) in tiles:
            crop = img.crop((x1, y1, x2, y2))
            results = model.predict(crop, conf=conf, verbose=False, save=False)
            r = results[0]
            if r.obb is None or len(r.obb) == 0:
                continue
            polys = r.obb.xyxyxyxy.cpu().numpy()   # local tile coords, shape (N,4,2)
            confs = r.obb.conf.cpu().numpy()
            clss = r.obb.cls.cpu().numpy()
            for poly, c, cl in zip(polys, confs, clss):
                global_poly = poly + np.array([x1, y1])   # shift into full-image coords
                all_detections.append({"poly": global_poly, "conf": float(c), "cls": int(cl)})

    return all_detections, (width, height)


def polygon_iou(poly1, poly2):
    p1, p2 = Polygon(poly1), Polygon(poly2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def rotated_nms(detections, iou_threshold=0.4):
    """Removes duplicate detections (same space caught by both grids/overlapping tiles), keeping the highest-confidence version."""
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if polygon_iou(best["poly"], d["poly"]) < iou_threshold]
    return keep


def draw_detections(image_path, detections, class_names, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        pts = [tuple(p) for p in d["poly"]]
        draw.polygon(pts, outline=(0, 0, 255), width=3)
        label = f"{class_names[d['cls']]} {d['conf']:.2f}"
        draw.text((pts[0][0], pts[0][1] - 14), label, fill=(0, 0, 255))
    img.save(out_path)
    return img
