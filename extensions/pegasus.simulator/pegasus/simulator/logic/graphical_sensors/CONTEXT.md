# Graphical Sensors Component

## Purpose
Camera and LiDAR sensors that use Isaac Sim's rendering pipeline for image-based sensor simulation.

## Inputs
- Vehicle stage prefix and camera prim path
- Camera configuration (resolution, FOV, update rate)

## Outputs
- Camera images (RGB, depth) via OmniGraph or direct API
- LiDAR point clouds (currently disabled)

## Dependencies
- graphs/ (ROS2CameraGraph for publishing camera data to ROS2 topics)

## Key Files
- `graphical_sensor.py` — Base `GraphicalSensor` abstract class
- `monocular_camera.py` — MonocularCamera implementation
- `lidar.py` — LiDAR sensor (currently commented out in `__init__.py`)

## MonocularCamera: Zoom
- `set_zoom(zoom: float)`: Scales intrinsic focal lengths (fx, fy) by the zoom factor while preserving the principal point (cx, cy). Recomputes and applies focal length, aperture, and clipping to the Isaac Sim camera.
- Minimum zoom is 1.0 (clamped). Original intrinsics stored in `_original_intrinsics`.
- Called by `ROS2Backend.zoom_callback()` via the `/{namespace}{id}/camera/zoom` ROS2 topic.

## Calling Contract
- `GraphicalSensor.initialize(vehicle)`: Called once after vehicle spawn.
- `GraphicalSensor.update(state, dt)`: Called every step by Vehicle.
