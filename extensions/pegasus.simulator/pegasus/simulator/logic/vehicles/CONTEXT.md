# Vehicles Component

## Purpose
Vehicle base class and Multirotor specialization. Defines the interface for spawning, updating, and controlling simulated vehicles. Composes sensors, dynamics, thrusters, and backends via pluggable configuration.

## Inputs
- USD file path for visual model
- Initial pose (position ENU, orientation quat `[qx, qy, qz, qw]`)
- MultirotorConfig: sensors list, dynamics, thrust curve, backends, graphical sensors
- Motor commands from backends (rotor speeds)

## Outputs
- Vehicle state per step: position, orientation, linear/angular velocity
- Forces and torques applied to articulation
- Sensor data forwarded to backends

## Dependencies
- sensors/ (composed via config)
- dynamics/ (composed via config)
- thrusters/ (composed via config)
- backends/ (composed via config)

## Key Files
- `vehicle.py` — Base Vehicle class (spawns USD, reads state, update loop)
- `multirotor.py` — Multirotor + MultirotorConfig (quadrotor specialization)
- `iris.py` — Iris 3DR quadrotor preset
- `typhoon_h480.py` — Typhoon H480 hexarotor preset

## Calling Contract
- `Vehicle.initialize(physics_sim_view)`: Called once after spawning.
- `Vehicle.update_sim(dt)`: Called every physics step. Updates state, sensors, backends, applies forces.
- `Vehicle.start() / stop()`: Lifecycle management.
