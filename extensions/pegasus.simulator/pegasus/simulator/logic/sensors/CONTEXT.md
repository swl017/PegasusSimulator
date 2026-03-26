# Sensors Component

## Purpose
Simulated sensor models that convert vehicle state into noisy sensor readings matching real hardware output. Used to provide realistic sensor data to autopilot backends.

## Inputs
- Vehicle `State`: position (ENU), orientation (quat), linear velocity, angular velocity
- Simulation timestep `dt`

## Outputs
- Sensor-specific readings (dict format consumed by backends):
  - IMU: acceleration `(3,)`, angular velocity `(3,)` (body frame, with noise + bias)
  - GPS: latitude, longitude, altitude, velocity (with noise)
  - Barometer: absolute pressure, temperature, altitude (with noise)
  - Magnetometer: magnetic field `(3,)` (body frame, with noise)

## Dependencies
None (standalone, plugged into Vehicle via config list).

## Key Files
- `sensor.py` — Base `Sensor` abstract class
- `imu.py` — IMU model (accelerometer + gyroscope with bias and noise)
- `gps.py` — GPS model (position + velocity with noise)
- `barometer.py` — Barometric pressure/altitude model
- `magnetometer.py` — Magnetic field model

## Calling Contract
- `Sensor.update(state, dt)`: Called by Vehicle every step. Returns sensor data dict.
- `Sensor.initialize(...)`: One-time setup with global coordinates.
