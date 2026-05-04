# Autonomous Carpark Patrol Pipeline

This repository contains the pipeline to process raw 3D LiDAR point clouds into 2D navigation maps, and allows for manual semantic annotation of parking spaces to generate precise GPS waypoints for an autonomous patrolling robot.

## Complete Workflow Guide

### Step 1: Generate the Maps (`main.py`)
This program turns your raw 3D LiDAR scan into flat 2D maps and a Google Earth file.
1. Place your raw `.ply` point cloud files into the `PLYInput/` folder.
2. Open your terminal and run: `python main.py`
3. The script will prompt you for a starting Latitude and Longitude. Find the rough center of your parking lot on Google Earth, copy the coordinates, and paste them right into the terminal (the script will automatically clean up the Google Earth text format).
4. **Output:** A new folder is created in `output/` containing:
   * `orthomosaic.png` (The visual top-down map)
   * `occupancy.png` (The collision map for the robot)
   * `[Name].kml` (The Google Earth file)

### Step 2: (Optional but Recommended) Align the Map perfectly
Since LiDAR scans rarely point perfectly North, your map might be slightly twisted compared to the real world.
1. Open the generated `.kml` file in **Google Earth Pro**.
2. Right-click the **Orthomosaic** layer in the sidebar and select **Properties** (Windows) / **Get Info** (Mac).
3. Use the green handles on the screen to drag, scale, and **rotate** the map until it perfectly matches the real-world satellite parking lines.
4. Click OK, then right-click the folder in the sidebar -> **Save Place As...** and overwrite your original `.kml` file in the `output/` folder.

### Step 3: Draw the Parking Spaces (`parking_annotator.py`)
This is where you tell the robot where the specific parking spots (targets) actually are.
1. Run: `python parking_annotator.py`
2. Click **Load Map Image** and select the `orthomosaic.png` from your `output/` folder.
3. Left-click to draw corners around a parking space. Right-click to finish the polygon. Repeat for all spaces.
4. Click **Save Annotations**.
5. **Output:** It generates an `orthomosaic_parking_spaces.json` file in the same folder. *(Note: These coordinates are currently just image pixels).*

### Step 4: Convert to Real-World GPS (`coordinate_converter.py`)
This final script bridges the gap between your drawn image and the real world.
1. Run: `python coordinate_converter.py`
2. It will automatically scan your `output/` folder.
3. It takes the perfect bounds from your `.kml` file, the image size, and your drawn annotations, and does the math to translate the pixels into real-world coordinates.
4. **Final Output:** It generates `orthomosaic_parking_spaces_gps.json`.

---

### The Final Result
You now have the two exact things your robot's Navigation Stack needs to patrol:
1. **The Map (`occupancy.png`)**: Tells the robot where it is safe to drive and where obstacles are.
2. **The Targets (`orthomosaic_parking_spaces_gps.json`)**: Gives the robot a list of precise GPS coordinates (and a calculated center waypoint) for every single parking space it needs to visit.
