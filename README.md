# LiDAR-to-2D Navigation Map Pipeline

This tool converts 3D LiDAR data (PLY files) into 2D navigation maps for car parking projects. It generates a top-down orthomosaic (preserving road markings) and an occupancy grid (detecting obstacles like pillars and walls).

## Features
- **PLY Loading**: Extracts per-vertex RGB color data.
- **Robust Cleaning**: Voxel downsampling and statistical outlier removal.
- **Ground Segmentation**: Uses RANSAC to identify the ground plane while preserving road markings.
- **Obstacle Detection**: Categorizes points between 0.1m and 2.5m as obstacles.
- **Georeferenced KML**: Exports maps as KML GroundOverlays for use in navigation software.

## Requirements
Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline with a `.ply` file:
```bash
python main.py --input path/to/cloud.ply --lat <origin_latitude> --lon <origin_longitude>
```

### Arguments
- `--input`: Path to the input `.ply` file (Required).
- `--lat`: Latitude of the local origin (Default: 37.7749).
- `--lon`: Longitude of the local origin (Default: -122.4194).
- `--voxel_size`: Size of the voxel for downsampling in meters (Default: 0.05).
- `--obs_height`: Minimum height above ground to be considered an obstacle (Default: 0.1).
- `--max_height`: Maximum height to consider, clips ceilings (Default: 2.5).
- `--resolution`: Map resolution in pixels per meter (Default: 20).
- `--output_dir`: Directory for output files (Default: output).

## Output
The pipeline generates:
1. `orthomosaic.png`: Top-down RGB image of the ground.
2. `occupancy.png`: 2D occupancy grid (White: Walkable, Black: Blocked).
3. `map.kml`: Georeferenced KML file containing the orthomosaic.
