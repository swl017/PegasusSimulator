# PegasusSimulator Logic Architecture

Core simulation logic for the Pegasus aerial vehicle simulator.

## Component Dependency Graph

```
┌──────────────────────────────────────────────────────────┐
│        PegasusInterface (Singleton Orchestrator)          │
│        Manages World, VehicleManager, global config       │
└──────────────────┬───────────────────────────────────────┘
                   │ creates & manages
                   ▼
          VehicleManager (Singleton)
                   │ tracks all spawned vehicles
                   ▼
┌──────────────────────────────────────────────────────────┐
│              Vehicle (Base Class)                          │
│              Composes all pluggable components             │
└──────────────────┬───────────────────────────────────────┘
                   │ configured via MultirotorConfig
    ┌──────────────┼──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
 sensors/     dynamics/     thrusters/     backends/
 (IMU,GPS,    (LinearDrag)  (Quadratic     (PX4Mavlink,
  Baro,Mag)                  ThrustCurve)   ArduPilot,
    │                                        ROS2)
    ▼
 graphical_sensors/    graphs/
 (MonocularCamera)     (ROS2CameraGraph)

Standalone:
  people/ (Person, PersonController, LinePersonController)
  └── PeopleManager (manages person actors)
```

## Directed Dependencies

```
Vehicle ──→ sensors, dynamics, thrusters, backends  (composed via config)
Multirotor ──→ Vehicle  (inheritance)
Iris, TyphoonH480 ──→ Multirotor  (concrete models)
PegasusInterface ──→ VehicleManager  (creates and holds reference)
graphs/ ──→ graphical_sensors/  (ROS2CameraGraph works with MonocularCamera)
```

All leaf components (sensors, dynamics, thrusters, backends) are standalone and pluggable.

## Data Flow Per Simulation Step

```
PegasusInterface.world physics step
       │
       ▼
Vehicle.update_sim(dt)
  ├─ Read state from USD stage (position, orientation, velocity, angular_velocity)
  ├─ Update sensors:
  │   ├─ IMU.update(state, dt) → acceleration, angular_velocity (+ noise)
  │   ├─ GPS.update(state, dt) → lat, lon, alt (+ noise)
  │   ├─ Barometer.update(state, dt) → pressure, altitude (+ noise)
  │   ├─ Magnetometer.update(state, dt) → magnetic field (+ noise)
  │   └─ GraphicalSensors.update() → camera images
  ├─ Send sensor data to backends:
  │   ├─ PX4MavlinkBackend.update_sensor_data(sensor_data)
  │   ├─ ArduPilotMavlinkBackend.update_sensor_data(sensor_data)
  │   └─ ROS2Backend.update_sensor_data(sensor_data)
  ├─ Receive motor commands from backends:
  │   └─ backend.input_reference() → rotor speeds
  ├─ Apply dynamics:
  │   ├─ ThrustCurve.thrust(rotor_speeds) → forces
  │   └─ Drag.update(state, dt) → drag force
  └─ Apply forces to articulation
```

## Key Data Containers

| Container | Location | Description |
|-----------|----------|-------------|
| `State` | `state.py` | Vehicle state: position, orientation (quat), velocity, angular velocity |
| `MultirotorConfig` | `vehicles/multirotor.py` | Composes sensors, dynamics, thrusters, backends, USD path |
| `BackendConfig` | `backends/backend.py` | Base config for communication backends |

## Component Isolation

**Standalone** (no internal dependencies):
sensors, dynamics, thrusters, backends, people, graphical_sensors

**Has dependencies**:
- vehicles → sensors, dynamics, thrusters, backends (composition)
- graphs → graphical_sensors
- interface → vehicle_manager → vehicles

## File Conventions

- Base classes: `sensor.py`, `backend.py`, `vehicle.py`, `drag.py`, `thrust_curve.py`
- Concrete implementations: `imu.py`, `px4_mavlink_backend.py`, `linear_drag.py`, etc.
- `CONTEXT.md` — Component routing contracts
- `__init__.py` — Public API exports
