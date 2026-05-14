import argparse
import sys
import os
import glob
from point_cloud_processor import PointCloudProcessor
from map_generator import MapGenerator
from kml_exporter import KMLExporter
from road_graph_extractor import RoadGraphExtractor
from json_exporter import JSONExporter


def process_single_file(input_path, args):
    """Processes a single .ply file through the pipeline."""
    filename = os.path.basename(input_path)
    file_basename = os.path.splitext(filename)[0]
    
    # Create unique output directory for this file
    output_dir = os.path.join(args.output_dir, file_basename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n>>> Processing: {input_path}")
    print(f">>> Output directory: {output_dir}")

    # 1. Processing
    processor = PointCloudProcessor(
        voxel_size=args.voxel_size,
        obstacle_height=args.obs_height,
        max_height=args.max_height
    )
    
    try:
        pcd = processor.load_ply(input_path)
        pcd = processor.clean_and_downsample(pcd)
        
        # Segment ground
        ground_inliers, non_ground, plane_model = processor.segment_ground(pcd)
        
        # Align everything to the detected plane
        pcd = processor.align_to_ground(pcd, plane_model)
        
        # Extract categories from aligned cloud
        ground_points, obstacle_points = processor.extract_obstacles(pcd)
        
    except Exception as e:
        print(f"Error during point cloud processing for {filename}: {e}")
        return

    # 2. Map Generation
    map_gen = MapGenerator(resolution=args.resolution)
    ortho, occ, bounds = map_gen.generate_maps(ground_points, obstacle_points)
    
    ortho_path = os.path.join(output_dir, "orthomosaic.png")
    occ_path = os.path.join(output_dir, "obstacle_occupancy.png")
    map_gen.save_maps(ortho, occ, bounds, ortho_path, occ_path)

    if getattr(args, 'manual_edit', False):
        print(f"\n[PAUSED] Maps saved to {output_dir}")
        print("You may now manually edit 'obstacle_occupancy.png' to block out parking spots (paint them black/0).")
        print("You may ALSO paint pure Blue squares (B:255, G:0, R:0) to manually mark Entrances/Exits!")
        input("Press Enter to continue graph extraction...")
        import cv2
        # Read as color to detect blue entrance markers
        occ_reloaded = cv2.imread(occ_path, cv2.IMREAD_COLOR)
        if occ_reloaded is not None:
            import numpy as np
            # Extract blue pixels (pure blue in BGR is [255, 0, 0])
            # We use a slight tolerance in case of anti-aliasing or slight brush feathering
            blue_mask = cv2.inRange(occ_reloaded, np.array([200, 0, 0]), np.array([255, 50, 50]))
            
            # Convert back to grayscale for the rest of the pipeline
            occ = cv2.cvtColor(occ_reloaded, cv2.COLOR_BGR2GRAY)
            print("Successfully loaded manually edited occupancy map.")
        else:
            print(f"Error: Could not reload {occ_path}. Using original.")
            blue_mask = None
    else:
        blue_mask = None

    # 3. Road Graph Extraction
    graph_ext = RoadGraphExtractor(min_lane_width=args.min_lane_width, pixels_per_meter=args.resolution)
    nodes, edges = graph_ext.extract_graph(occ, output_dir, blue_mask)

    # 4. JSON Export (Cartesian)
    json_exp = JSONExporter(output_dir)
    json_path = json_exp.export_graph(nodes, edges, bounds, args.resolution, filename=f"{file_basename}_graph.json")

    # 5. KML Export (Optional/Secondary)
    exporter = KMLExporter(base_lat=args.lat, base_lon=args.lon)
    kml_path = os.path.join(output_dir, f"{file_basename}.kml")
    exporter.export_all(
        ortho_path=ortho_path,
        bounds=bounds,
        nodes=nodes,
        edges=edges,
        resolution=args.resolution,
        occ_path=occ_path,
        output_kml=kml_path
    )


    print(f"Finished processing {filename}. Results in {output_dir}")

import re

def parse_coordinate(coord_str):
    """Parses a coordinate string which could be a float or DMS format."""
    coord_str = str(coord_str).strip()
    
    # Try direct float conversion first
    try:
        return float(coord_str)
    except ValueError:
        pass
        
    # Clean up common copy-paste errors from Google Earth
    # e.g. "36°54 32.03S 174948 35.90"
    coord_str = coord_str.upper().replace('"', '').replace("'", ' ')
    coord_str = re.sub(r'[°º]', ' ', coord_str)
    
    # Extract numbers and hemisphere
    # Matches: "36 54 32.03 S" or "36 54 32.03S"
    match = re.search(r'(\d+)\s+(\d+)\s+([\d\.]+)\s*([NSEW])', coord_str)
    
    if match:
        deg, min_val, sec, hemi = match.groups()
        dd = float(deg) + (float(min_val) / 60) + (float(sec) / 3600)
        if hemi in ['S', 'W']:
            dd = -dd
        return round(dd, 8)
        
    raise ValueError(f"Could not parse coordinate: {coord_str}")

def get_interactive_args(args):
    """Prompts the user for arguments if they are not provided via CLI."""
    print("\n--- Debugger Mode: Interactive Configuration ---")
    
    # Input Path
    default_input = args.input or "PLYInput"
    val = input(f"Enter input file or folder path [{default_input}]: ").strip()
    if val: args.input = val
    else: args.input = default_input

    # Coordinates
    while True:
        val = input(f"Enter base Latitude (Float or DMS e.g. 36°54 32S) [{args.lat}]: ").strip()
        if not val:
            break
        try:
            args.lat = parse_coordinate(val)
            print(f"  -> Parsed Latitude: {args.lat}")
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")

    while True:
        val = input(f"Enter base Longitude (Float or DMS e.g. 174°48 35E) [{args.lon}]: ").strip()
        if not val:
            break
        try:
            # Handle potential typos like '174948' instead of '174°48'
            if len(val) > 10 and ' ' in val and not any(c in val for c in ['°', 'N', 'S', 'E', 'W']):
                print("Note: Appending 'E' to your input to assist with parsing...")
                val += 'E'
            
            # Very hacky fix for "174948 35.90"
            if '9' in val and '°' not in val:
                val = val.replace('9', ' ', 1)

            args.lon = parse_coordinate(val)
            print(f"  -> Parsed Longitude: {args.lon}")
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")

    val = input("Pause for manual occupancy edit? (y/N) [N]: ").strip().lower()
    if val == 'y':
        args.manual_edit = True

    print("------------------------------------------------\n")
    return args

def main():
    parser = argparse.ArgumentParser(description="LiDAR-to-2D Navigation Map Pipeline")
    parser.add_argument("--input", type=str, default="PLYInput", help="Path to input .ply file or directory")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size (meters)")
    parser.add_argument("--obs_height", type=float, default=0.1, help="Min height for obstacles (meters)")
    parser.add_argument("--max_height", type=float, default=2.0, help="Max height for points (meters)")
    parser.add_argument("--resolution", type=float, default=70, help="Map resolution (pixels/meter)")
    parser.add_argument("--min_lane_width", type=float, default=2.5, help="Minimum lane width (meters)")
    parser.add_argument("--lat", type=float, default=37.7749, help="Base Latitude (origin)")
    parser.add_argument("--lon", type=float, default=-122.4194, help="Base Longitude (origin)")
    parser.add_argument("--output_dir", type=str, default="output", help="Base directory for output files")
    parser.add_argument("--manual_edit", action="store_true", help="Pause to allow manual editing of obstacle_occupancy.png before graph extraction")
    
    # Check if run with no arguments (typical for debugger launch)
    if len(sys.argv) == 1:
        args = parser.parse_args([]) # Get defaults
        args = get_interactive_args(args)
    else:
        args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Make output_dir relative to script directory if it's not an absolute path
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determine files to process
    input_path = args.input
    
    # If the path doesn't exist, try looking in the default folder (PLYInput)
    if not os.path.exists(input_path):
        script_rel_path = os.path.join(script_dir, input_path)
        alt_path = os.path.join(script_dir, "PLYInput", input_path)
        
        if os.path.exists(script_rel_path):
            input_path = script_rel_path
        elif os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"Error: Input path '{args.input}' does not exist (checked CWD, script dir, and PLYInput/).")
            sys.exit(1)

    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.ply"))
        if not files:
            print(f"No .ply files found in directory: {input_path}")
            sys.exit(0)
        print(f"Found {len(files)} files in {input_path}")
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory.")
        sys.exit(1)

    # Batch process
    for f in files:
        process_single_file(f, args)

    print("\nAll tasks finished successfully.")

if __name__ == "__main__":
    main()
