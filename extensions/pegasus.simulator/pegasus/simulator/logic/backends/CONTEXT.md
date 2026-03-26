# Backends Component

## Purpose
Communication interfaces between the simulated vehicle and external autopilot software (PX4, ArduPilot) or middleware (ROS2). Backends receive sensor data from vehicles and return motor commands.

## Inputs
- Sensor data from Vehicle.update_sim(): IMU, GPS, barometer, magnetometer readings
- Vehicle state: position, orientation, velocity

## Outputs
- Motor commands (rotor speeds) via `input_reference()`
- MAVLink messages (PX4/ArduPilot) or ROS2 topics

## Dependencies
None (standalone, plugged into Vehicle via config).

## Key Files
- `backend.py` — Base `Backend` and `BackendConfig` abstract classes
- `px4_mavlink_backend.py` — PX4 MAVLink communication (HIL_SENSOR, HIL_GPS, HIL_STATE_QUATERNION)
- `ardupilot_mavlink_backend.py` — ArduPilot MAVLink communication
- `ros2_backend.py` — ROS2 topic publishing (optional, requires ROS2 installation)

## ROS2Backend: Zoom Control
- Config flag: `"sub_zoom": True` (default True)
- Subscribes to `/{namespace}{id}/camera/zoom` (`std_msgs/Float64`) in `start()` — not in `initialize_subscribers()`, because `self._vehicle` is only available after the backend is attached to the vehicle.
- `zoom_callback()` iterates `self._vehicle._graphical_sensors`, calls `MonocularCamera.set_zoom(zoom)` on each camera.
- Zoom value: 1.0 = no zoom (minimum), 2.0 = 2x, etc.

## Calling Contract
- `update_sensor_data(state, sensor_data)`: Called by Vehicle every step. Sends sensor readings to autopilot.
- `input_reference()`: Called by Vehicle to get current motor commands from autopilot.
- `start() / stop()`: Lifecycle management (open/close MAVLink connections).
