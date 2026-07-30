# Two-Repo Go2 Carpark Localization Plan

Build a map-localized patrol pipeline that bridges the upstream car-park perception repo with the Unitree Go2 control repo. The upstream repo already turns LiDAR/PLY data into a 2D orthomosaic, occupancy grid, road graph, and parking-space annotations, so the Go2 repo should consume that map product, estimate its own pose on the map, and drive to waypoints with move commands. Manual controller support is out of scope except as a test aid; the robot should be commanded directly to waypoint goals, not teleoperated by a human.

## Steps

1. Lock the cross-repo data contract.
   - Define one shared map frame for both repos, including origin, axis direction, resolution, and units.
   - Decide which output artifacts from the car-park repo are authoritative for the robot: occupancy grid, orthomosaic, road graph JSON, parking-space JSON, and waypoint JSON.
   - Make the waypoint schema explicit so the Go2 repo can consume target positions and tolerances without needing to know how they were generated.
   - Depends on the existing pipeline outputs in the car-park repo and the waypoint file in Smart Parking Park_go2_waypoints.json.

2. Connect the car-park pipeline to a robot-usable navigation map.
   - Keep the upstream repo as the source of map generation: LiDAR/PLY processing -> map generation -> road graph extraction -> annotation -> waypoint export.
   - Use the orthomosaic for human review and parking-space extraction if needed, but treat the occupancy/graph output as the main navigation substrate.
   - Add or preserve export formats that the Go2 side can load directly, including a robot-frame map and a waypoint/graph file.
   - If the existing outputs are image-centric, add a translation step so image-space detections are converted back into map coordinates.
   - Depends on step 1 and the upstream repo files such as `map_generator.py` and `road_graph_extractor.py`.

3. Build the robot-side localization layer.
   - Implement a pose estimator in the Go2 repo that can answer x, y, yaw, and confidence relative to the shared map.
   - Start with lidar-based localization or scan matching as the primary method because the problem is “where am I in this scan/map?”
   - Optionally use camera data as a secondary signal for drift correction or recovery, but do not make camera VSLAM the only dependency unless the map design forces it.
   - Keep the localization module separate from motion control so the robot can be tested for pose estimation without moving.
   - Depends on step 1 and the available sensor/state plumbing in the Go2 SDK examples.

4. Replace manual control with waypoint navigation commands.
   - Refactor the Go2 patrol script so it sends move commands toward waypoint targets rather than relying on a human controller.
   - Make arrival detection depend on estimated pose and waypoint tolerance, not on fixed timing alone.
   - Keep a dry-run mode that simulates pose estimates and goal convergence for testing, but remove any assumption that a human is driving the robot.
   - The controller should expose a simple loop: get pose, compare to target, compute motion command, repeat until within tolerance.
   - Depends on steps 1 and 3, and the current patrol entrypoint in `run_go2_patrol.py`.

5. Add map-to-waypoint generation and validation.
   - Continue using the upstream repo’s parking-space and graph tooling to create or validate waypoint sets.
   - Ensure waypoint generation preserves enough metadata for the Go2 side to know what each waypoint means, such as entrance, turn, parking lane, or loop point.
   - Validate that generated waypoints land in reachable regions of the occupancy map before they are handed to the robot.
   - Depends on steps 1 and 2, and may be developed in parallel with step 3 once the shared frame is fixed.

6. Wire the Go2 SDK interfaces needed for localization and motion.
   - Reuse the existing Unitree SDK patterns for sensor access, client initialization, and motion commands.
   - Subscribe to or poll the state data required by the localization layer instead of treating motion commands as the source of truth.
   - Keep obstacle avoidance and safety behaviors as protective layers, but not as the localization mechanism.
   - Depends on the controller architecture chosen in steps 3 and 4.

7. Verify end-to-end behavior in layers.
   - First validate the upstream repo produces a stable map and waypoint set from a sample car-park scan.
   - Then validate the Go2 repo can load that map and estimate pose against it in offline or dry-run mode.
   - Finally verify the robot can be commanded to move to map waypoints using pose feedback and stop at tolerance.
   - Include a negative test where pose confidence drops and the robot halts instead of continuing blindly.

## Relevant Files

- `README.md` - upstream pipeline description and outputs.
- `map_generator.py` - creates orthomosaic and occupancy products.
- `road_graph_extractor.py` - turns occupancy into a route graph.
- `parking_annotator.py` - parking-space and entrance annotation UI.
- `unified_pipeline_gui.py` - upstream orchestration of the full map pipeline.
- `run_go2_patrol.py` - Go2 patrol entrypoint that should become pose-driven.
- `Go2KeyboardController.py` - test-only manual control harness; not part of the final operating path.
- `Smart Parking Park_go2_waypoints.json` - current waypoint contract to preserve or evolve.
- `sport_client.py` - Go2 motion API.
- `_SportModeState_.py` - pose/state fields available from the SDK.
- `camera_opencv.py` - optional camera acquisition reference.

## Verification

1. Confirm the upstream repo can produce a stable map/graph/waypoint output from a sample scan.
2. Confirm the Go2 repo can ingest that output and compute pose relative to the same frame.
3. Confirm the waypoint controller issues move commands from pose error, not from manual input.
4. Confirm dry-run mode simulates localization error and convergence without a live robot.
5. Confirm a low-confidence localization case stops the robot safely.

## Decisions

- The upstream repo remains the map-generation source of truth.
- The Go2 repo remains the execution and localization side.
- Manual controller code is test-only and should not be part of the final navigation flow.
- The first implementation should prefer lidar/map localization over camera-only VSLAM, because the task is to know where the robot is on the generated car-park map.

## Further Considerations

1. Do you want the first integration milestone to target occupancy-grid localization only, or occupancy-grid plus orthomosaic alignment?
2. Do you want waypoint generation to stay in the upstream repo, or should the Go2 repo also be able to derive simple fallback waypoints from the map directly?
3. Do you want the robot to stop immediately on low confidence, or attempt a recovery scan before halting?