import argparse
import sys
import os
import glob
from point_cloud_processor import PointCloudProcessor
from map_generator import MapGenerator
from kml_exporter import KMLExporter

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
    occ_path = os.path.join(output_dir, "occupancy.png")
    map_gen.save_maps(ortho, occ, ortho_path, occ_path)

    # 3. KML Export
    exporter = KMLExporter(base_lat=args.lat, base_lon=args.lon)
    kml_path = os.path.join(output_dir, f"{file_basename}.kml")
    exporter.export_orthomosaic(ortho_path, bounds, occ_path=occ_path, output_kml=kml_path)

    print(f"Finished processing {filename}. Results in {output_dir}")

def get_interactive_args(args):
    """Prompts the user for arguments if they are not provided via CLI."""
    print("\n--- Debugger Mode: Interactive Configuration ---")
    
    # Input Path
    default_input = args.input or "PLYInput"
    val = input(f"Enter input file or folder path [{default_input}]: ").strip()
    if val: args.input = val
    else: args.input = default_input

    # Coordinates
    val = input(f"Enter base Latitude [{args.lat}]: ").strip()
    if val: args.lat = float(val)

    val = input(f"Enter base Longitude [{args.lon}]: ").strip()
    if val: args.lon = float(val)

    print("------------------------------------------------\n")
    return args

def main():
    parser = argparse.ArgumentParser(description="LiDAR-to-2D Navigation Map Pipeline")
    parser.add_argument("--input", type=str, default="PLYInput", help="Path to input .ply file or directory")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size (meters)")
    parser.add_argument("--obs_height", type=float, default=0.1, help="Min height for obstacles (meters)")
    parser.add_argument("--max_height", type=float, default=2.0, help="Max height for points (meters)")
    parser.add_argument("--resolution", type=float, default=70, help="Map resolution (pixels/meter)")
    parser.add_argument("--lat", type=float, default=37.7749, help="Base Latitude (origin)")
    parser.add_argument("--lon", type=float, default=-122.4194, help="Base Longitude (origin)")
    parser.add_argument("--output_dir", type=str, default="output", help="Base directory for output files")
    
    # Check if run with no arguments (typical for debugger launch)
    if len(sys.argv) == 1:
        args = parser.parse_args([]) # Get defaults
        args = get_interactive_args(args)
    else:
        args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determine files to process
    input_path = args.input
    
    # If the path doesn't exist, try looking in the default folder (PLYInput)
    if not os.path.exists(input_path):
        alt_path = os.path.join("PLYInput", input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"Error: Input path '{args.input}' does not exist (checked root and PLYInput/).")
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
