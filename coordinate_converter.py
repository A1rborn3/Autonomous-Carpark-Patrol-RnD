import os
import json
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

def extract_bounds_from_kml(kml_path):
    """Parses a KML file to extract the bounding box coordinates."""
    # KML uses namespaces
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Find the LatLonBox
    # Orthomosaic overlay is usually the first GroundOverlay
    lat_lon_box = root.find('.//kml:GroundOverlay/kml:LatLonBox', ns)
    
    if lat_lon_box is None:
        raise ValueError(f"Could not find LatLonBox in {kml_path}")
        
    north = float(lat_lon_box.find('kml:north', ns).text)
    south = float(lat_lon_box.find('kml:south', ns).text)
    east = float(lat_lon_box.find('kml:east', ns).text)
    west = float(lat_lon_box.find('kml:west', ns).text)
    
    return {'north': north, 'south': south, 'east': east, 'west': west}

def convert_pixels_to_gps(points, bounds, img_width, img_height):
    """Converts a list of (x,y) pixel coordinates to (lat, lon) GPS coordinates."""
    gps_points = []
    for x, y in points:
        # Linear interpolation
        # x is from 0 to img_width -> west to east
        lon = bounds['west'] + (x / img_width) * (bounds['east'] - bounds['west'])
        
        # y is from 0 to img_height -> north to south
        # (y=0 is top/north, y=img_height is bottom/south)
        lat = bounds['north'] - (y / img_height) * (bounds['north'] - bounds['south'])
        
        gps_points.append((lat, lon))
    return gps_points

def compute_centroid(points):
    """Computes the center (lat, lon) of a polygon."""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (sum(lats) / len(lats), sum(lons) / len(lons))

def process_directory(directory):
    print(f"Processing directory: {directory}")
    
    # 1. Find KML file
    kml_files = [f for f in os.listdir(directory) if f.endswith('.kml')]
    if not kml_files:
        print(f"Error: No .kml file found in {directory}")
        return
    kml_path = os.path.join(directory, kml_files[0])
    
    # 2. Find orthomosaic image
    img_path = os.path.join(directory, 'orthomosaic.png')
    if not os.path.exists(img_path):
        print(f"Error: orthomosaic.png not found in {directory}")
        return
        
    # 3. Find JSON annotations
    json_path = os.path.join(directory, 'orthomosaic_parking_spaces.json')
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Have you annotated this map yet?")
        return

    # Extract bounds
    bounds = extract_bounds_from_kml(kml_path)
    print(f"Found GPS bounds: {bounds}")
    
    # Get image dimensions
    with Image.open(img_path) as img:
        img_width, img_height = img.size
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Process annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    parking_spaces = data.get('parking_spaces', [])
    gps_parking_spaces = []
    
    for space in parking_spaces:
        space_id = space['id']
        pixel_points = space['points']
        
        gps_points = convert_pixels_to_gps(pixel_points, bounds, img_width, img_height)
        centroid = compute_centroid(gps_points)
        
        gps_parking_spaces.append({
            'id': space_id,
            'points': gps_points,
            'centroid': centroid
        })
        
    # Save output
    out_data = {
        'source_image': 'orthomosaic.png',
        'kml_reference': kml_files[0],
        'parking_spaces': gps_parking_spaces
    }
    
    out_path = os.path.join(directory, 'orthomosaic_parking_spaces_gps.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=4)
        
    print(f"Successfully converted {len(gps_parking_spaces)} parking spaces.")
    print(f"Saved to: {out_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert pixel annotations to GPS coordinates")
    parser.add_argument("--dir", type=str, help="Directory containing the map files (kml, png, json)")
    
    args = parser.parse_args()
    
    if args.dir:
        process_directory(args.dir)
    else:
        # If no dir is provided, run on all subdirectories in 'output/'
        output_dir = 'output'
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                sub_dir = os.path.join(output_dir, item)
                if os.path.isdir(sub_dir):
                    process_directory(sub_dir)
        else:
            print("No --dir provided and 'output/' directory not found.")

if __name__ == "__main__":
    main()
