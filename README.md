# Autonomous Carpark Patrol R&D

This project turns 3D LiDAR point clouds into a navigable 2D map for autonomous carpark operations. The workflow starts with a PLY scan, builds a metric orthomosaic and obstacle map, lets a human annotate parking spaces and entrances/exits, and then extracts a road graph or patrol route for robot navigation and export.

It is designed around a desktop GUI workflow rather than a single command-line script. The main entry point is the application in `main_gui.py`, which coordinates three stages:

1. Map generation from LiDAR
2. Parking annotator for parking spaces and access points
3. Robot route plotting and graph export

## What the project does

- Loads `.ply` point cloud data and extracts ground and obstacle points
- Creates an orthomosaic image of the traversable surface
- Builds an occupancy map for navigation and graph extraction
- Supports manual parking annotation, including parking-space polygons and entrance/exit markers
- Extracts a road graph from the occupancy image
- Exports route data to JSON and Unitree Go2 patrol files
- Produces route visualizations and route metadata compatible with downstream navigation tools

## Project structure

- `main_gui.py` — main application shell with notebook tabs for the full workflow
- `unified_pipeline_gui.py` — LiDAR map generation pipeline and graph extraction UI
- `parking_annotator.py` — carpark annotation tool for parking spaces and entrances/exits
- `robot_line_plotter.py` — manual route plotting and export interface
- `point_cloud_processor.py` — PLY loading, cleaning, ground segmentation, and obstacle extraction
- `map_generator.py` — 2D occupancy and orthomosaic generation and metadata export
- `road_graph_extractor.py` — automated graph extraction from occupancy maps
- `Graph_Proccessing.py` — post-processing for graph-based patrol route generation
- `json_exporter.py` — graph export to JSON
- `unitree_exporter.py` — Unitree Go2 waypoint and patrol script export
- `inference.py` and `inference_config.py` — inference-related annotation support
- `output/` — generated map and route outputs
- `model/` — model weights stored for object detection tasks
- `PLYInput/` — example or working LiDAR inputs

## Requirements

The project uses Python 3.11 and the dependencies in `requirements.txt`.

Install them with:

```bash
python -m pip install -r requirements.txt
```

Key packages include:

- Open3D
- OpenCV
- NumPy / SciPy
- scikit-image
- Pillow
- Matplotlib
- Ultralytics
- Shapely

## Running the application

Launch the full GUI workflow:

```bash
python main_gui.py
```

This opens a tabbed interface with these stages:

### 1. Step 1: Map Generation

- Choose a PLY input
- Adjust pipeline parameters such as:
  - voxel size
  - minimum obstacle height
  - maximum height
  - map resolution
  - minimum lane width
- Run the pipeline to generate:
  - orthomosaic image
  - occupancy map
  - map metadata JSON
  - a dedicated output folder named after the input file

### 2. Step 2: Carpark Annotator

- Load the generated orthomosaic
- Draw parking spaces and entrance/exit polygons
- Assign parking types such as Regular, Handicap, or 60 Mins Max
- Save annotations
- The tool writes parking-space annotation files alongside the map output

### 3. Step 3: Navigation Plotter

- Load the map image
- Plot a manual robot patrol route line
- Save route JSON
- Export waypoints directly for manual navigation
- Run automated graph extraction from the occupancy map to generate a graph and route data

## Typical workflow

1. Collect a LiDAR point cloud in `.ply` format
2. Open the app via `python main_gui.py`
3. In the Map Generation tab, select the PLY and run the pipeline
4. Confirm the generated map in the annotator tab
5. Add parking spaces and entrance/exit annotations
6. Switch to the Navigation Plotter and draw or extract the route
7. Export the finalized route graph and Unitree Go2 patrol files from the output folder

## Output artifacts

Each processed scan creates a folder under `output/` using the source file name, for example:

```text
output/
  my_site/
    orthomosaic.png
    obstacle_occupancy.png
    merged_occupancy.png
    map_metadata.json
    my_site_graph.json
    Manual_Output/
    Automated_Output/
```

Common generated files include:

- `orthomosaic.png` — RGB orthomosaic of the ground surface
- `obstacle_occupancy.png` — obstacle map used for graph extraction
- `merged_occupancy.png` — occupancy map with annotations merged in
- `map_metadata.json` — metric bounds and resolution metadata
- `*_graph.json` — graph export in cartesian coordinates
- `run_go2_patrol.py` — exported execution script for robot patrol navigation
- `*_final_route.png` — visualized route overlay image

## Notes

- The GUI is the intended user interface for this project and is the most accurate description of the current pipeline.
- The older command-line usage shown in earlier versions of this repository is no longer the primary interface.
- If you are working with a dataset, the workflow is best executed through the GUI so that annotations, route definitions, and exported graphs remain synchronized with the generated map metadata.

## License

This project is intended for research and autonomous navigation development use within the repository context. Check the repository status and any project-specific licensing terms before redistribution or commercial deployment.

