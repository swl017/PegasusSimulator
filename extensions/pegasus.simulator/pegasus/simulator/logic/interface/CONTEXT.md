# PegasusInterface Component

## Purpose
Singleton orchestrator that manages the Isaac Sim World, VehicleManager, and global simulation configuration (PX4/ArduPilot paths, global coordinates).

## Inputs
- World settings (physics dt, rendering dt, stage units)
- Global coordinates (latitude, longitude, altitude) from `config/configs.yaml`
- PX4/ArduPilot installation paths from config

## Outputs
- Configured `World` instance for simulation
- VehicleManager for spawning and tracking vehicles
- Environment loading (USD stages)

## Dependencies
- `vehicle_manager.py` (creates and holds VehicleManager singleton)
- `params.py` (default settings, environment paths)
- `config/configs.yaml` (runtime configuration)

## Key Files
- `pegasus_interface.py` — Singleton class with thread-safe initialization
- `../vehicle_manager.py` — VehicleManager singleton (tracks spawned vehicles)
- `../state.py` — State dataclass (position, orientation, velocity, angular velocity)
- `../rotations.py` — Rotation utilities

## Calling Contract
- `PegasusInterface()`: Thread-safe singleton via `__new__` + Lock. Safe to call multiple times.
- `load_environment(usd_path)`: Loads a USD stage. Call once per session.
- `load_asset(usd_asset, stage_prefix)`: Spawns an asset into the current stage.
- `set_world_settings(physics_dt, rendering_dt)`: Call before `load_environment()`.
