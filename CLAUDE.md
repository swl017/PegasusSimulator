# PegasusSimulator — Isaac Sim Extension for Aerial Vehicle Simulation

## Architecture
See [ARCHITECTURE.md](extensions/pegasus.simulator/pegasus/simulator/logic/ARCHITECTURE.md) for component dependency graph and data flow.

## Session Workflow Protocol

At the **START** of each session:
- Read `doc/active/feature_list.json` and `doc/active/progress.txt`
- Read `ARCHITECTURE.md` for component boundaries

At the **END** of each session:
- Append to `doc/active/progress.txt`: what was done, what's next
- Update `doc/active/feature_list.json` if any feature status changed
- Update the relevant component's `CONTEXT.md` if its interface changed
- Update `ARCHITECTURE.md` if component dependencies changed

## Component Navigation
Each component under `logic/` has a `CONTEXT.md` describing its purpose, inputs, outputs, dependencies, and key files. Read the relevant `CONTEXT.md` before working on a component.

## Folder Semantics
```
PegasusSimulator/
├── CLAUDE.md                          # This file (session workflow)
├── doc/
│   ├── active/                        # Multi-session tracking (feature_list.json, progress.txt)
│   ├── backlog/                       # Low-urgency future tasks
│   └── spec/                          # Spec docs
├── extensions/pegasus.simulator/      # Main Isaac Sim extension
│   ├── config/                        # extension.toml, configs.yaml
│   ├── pegasus/simulator/
│   │   ├── logic/                     # Core source code
│   │   │   ├── ARCHITECTURE.md        # Component dependency graph
│   │   │   ├── interface/             # PegasusInterface singleton
│   │   │   ├── vehicles/              # Vehicle base + Multirotor + models
│   │   │   ├── backends/              # PX4, ArduPilot, ROS2
│   │   │   ├── sensors/               # IMU, GPS, Barometer, Magnetometer
│   │   │   ├── graphical_sensors/     # Camera, LiDAR
│   │   │   ├── dynamics/              # Drag models
│   │   │   ├── thrusters/             # Thrust curve models
│   │   │   ├── graphs/                # OmniGraph computation graphs
│   │   │   └── people/                # Pedestrian simulation
│   │   ├── parser/                    # YAML config parsing
│   │   ├── ui/                        # GUI widgets
│   │   └── assets/                    # USD models, worlds
│   └── setup.py
├── examples/                          # Standalone example scripts
├── launch/                            # Isaac Sim launch scripts
├── docs/                              # Sphinx documentation
└── template_ros2/                     # ROS2 AI workflow templates
```

## Key Entry Points
- **Extension**: `pegasus/simulator/extension.py` — Isaac Sim extension entry
- **API**: `pegasus/simulator/logic/interface/pegasus_interface.py` — singleton manager
- **Config**: `config/configs.yaml` — PX4/ArduPilot paths, global coordinates
- **Params**: `pegasus/simulator/params.py` — default world settings, environment paths

## Conventions
- **Quaternion**: `[qx, qy, qz, qw]` (Hamilton, scalar-last) — note: differs from Isaac Lab's `wxyz`
- **Coordinate frame**: ENU (East-North-Up) for positions
- **Singleton pattern**: PegasusInterface, VehicleManager — use `__new__` with Lock
- **Plugin architecture**: Vehicles compose sensors, dynamics, thrusters, and backends via config

## Related Projects
- **`iris_ma6`**: A multi-agent reinforment learning environment for active triangulation `/home/usrg/IsaacPX4/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6`
- **`mas`**: A ROS2 system for deploying the learnt policy, sim-to-real