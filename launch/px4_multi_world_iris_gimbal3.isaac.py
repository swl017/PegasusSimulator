#!/usr/bin/env python
"""
| File: 2_px4_multi_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with multiple vehicles, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import argparse
import carb
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args, _ = parser.parse_known_args()

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": args.headless})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
# Auxiliary scipy and numpy modules
import os.path
from scipy.spatial.transform import Rotation
import numpy as np

import asyncio
import carb
import omni.ext
import omni.graph.core as og
import isaacsim.ros2.bridge as bridge

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Plane with Light"])

        # Load the Flight aesthetic scene (Y-up USD, rotated to Z-up)
        # self.load_flight_scene(
        #     usd_path="/home/usrg/IsaacPX4/world/Flight/Flight_original.usd",
        #     scale=0.001,
        #     offset=(0.0, 0.0, 0.0),
        # )

        self.world_offset_x, self.world_offset_y, self.world_offset_z = 0,0,0
        # self.world_offset_x = 0.0
        # self.world_offset_y = 0.0
        # self.world_offset_z = 0.0
        # self.world_offset_x = -606344.799574
        # self.world_offset_y = -1586.657862
        # self.world_offset_z = 1395514.256701
        # self.world.reset()
        asyncio.ensure_future(self.create_simulation_time_graph())
        self.create_landmarks()

        # Camera resolution preset (mirrors /home/usrg/mas/src/tmux/drone.tmuxp.yaml
        # and simdrone*.tmuxp.yaml). Override per-launch with `CAMERA_RES=low|mid|high`.
        # Lens is the same across presets — smaller resolutions are sensor crops of
        # the 1920×1080 reference: fx,fy held constant from calibration; cx,cy = w/2, h/2.
        #   low  →  640× 360,  YOLO engine dronecop9-2-384x640
        #   mid  →  960× 540,  YOLO engine dronecop9-2-544x960
        #   high → 1920×1080,  YOLO engine dronecop9-2-1088x1920
        _camera_res = os.environ.get("CAMERA_RES", "high")
        self._camera_res = _camera_res   # kept for the boot label (see write_boot_label)
        if _camera_res == "low":
            self._cam_width, self._cam_height = 640, 360
        elif _camera_res == "mid":
            self._cam_width, self._cam_height = 960, 540
        elif _camera_res == "high":
            self._cam_width, self._cam_height = 1920, 1080
        else:
            raise ValueError(f"unknown CAMERA_RES={_camera_res} (use low|mid|high)")

        # Intrinsics from 2026-04-17/1x calibration (SIYI A8 mini, 1x zoom, 1920×1080):
        #   /home/usrg/mas/datasets/camera_calibration/2026-04-17/1x/intrinsics_summary.json
        # Same-lens assumption → fx, fy constant across resolution presets; cx, cy snap
        # to image center.
        _fx_cal = 1053.044591
        _fy_cal = 1052.905959
        self._cam_matrix = [[_fx_cal, 0, self._cam_width / 2],
                            [0, _fy_cal, self._cam_height / 2],
                            [0, 0, 1]]
        print(f"[PegasusApp] CAMERA_RES={_camera_res}  {self._cam_width}x{self._cam_height}  "
              f"fx={_fx_cal:.2f} fy={_fy_cal:.2f} cx={self._cam_width/2:.1f} cy={self._cam_height/2:.1f}")

        # Derived USD camera parameters — same lens → focal_length constant; aperture
        # scales with resolution so rendered fx_px = focal*width/h_aperture = _fx_cal.
        # Value matches iris_ma6 training env (iris_ma_env6_test_cfg.py): focal_length=11.493,
        # horizontal_aperture=20.955 at 1920×1080 → fx_px = 11.493*1920/20.955 = 1053.04.
        self._cam_focal_length = 11.493  # mm (matches iris_ma6 training PinholeCameraCfg)
        self._cam_h_aperture = self._cam_focal_length * self._cam_width / _fx_cal
        self._cam_v_aperture = self._cam_focal_length * self._cam_height / _fy_cal

        self.namespace = "px4_"
        self.vehicles = []
        # RAL ticket 021 coop_2obs: NUM_VEHICLES=4 adds px4_4 (second flown observer,
        # same camera+gimbal factory). Default 3 keeps every existing session unchanged.
        num_vehicles = int(os.environ.get("NUM_VEHICLES", "3"))
        print(f"[PegasusApp] NUM_VEHICLES={num_vehicles}")

        # RAL ticket 019 revisit2 W0f — CAMERA_VEHICLES: which vehicle ids get a camera.
        # The wall clock of a sweep is set by Isaac RENDERING camera sensors, not by ROS-side
        # fusion CPU, and this factory attaches a 1920x1080 @ 25 Hz MonocularCamera with
        # pub_graphical_sensors:True to EVERY vehicle — including ones whose session starts
        # no camera/YOLO window. In the coop_1obs layout px4_3 (the TARGET) is rendered and
        # published for NO consumer (nothing in `mas` subscribes /px4_3/camera/color/*), so
        # excluding it drops the rendered camera count 3 -> 2, a third off the dominant cost.
        # Default = ALL ids, so every pre-existing session is byte-for-byte unchanged.
        #   CAMERA_VEHICLES="1,2"   -> only px4_1 and px4_2 carry cameras
        # Deliver it via the tmuxp `environment:` block or `tmux setenv -g` and read back with
        # `tmux show-environment -g`: `VAR=... tmuxp load` is silently dropped whenever a tmux
        # server exists, and on this box an idle `keepalive` session keeps one alive always.
        _cam_env = os.environ.get("CAMERA_VEHICLES", "").strip()
        if _cam_env:
            self._camera_ids = {int(x) for x in _cam_env.split(",") if x.strip()}
        else:
            self._camera_ids = set(range(1, num_vehicles + 1))
        print(f"[PegasusApp] CAMERA_VEHICLES={sorted(self._camera_ids)} "
              f"({len(self._camera_ids)} of {num_vehicles} vehicles rendered)")

        for i in range(num_vehicles):
            self.vehicle_factory(i+1, gap_x_axis=1.0)

        # RAL ticket 055 S21 — DISABLE_VEHICLE_COLLISION: stop vehicles colliding with
        # EACH OTHER (ground collision is untouched).
        #
        # WHY. The engagement harness runs every trial to TRUE closest-point-of-approach
        # and deliberately "never terminates on a radius crossing", so min_range is an
        # uncensored CPA that stays scorable at any kill radius
        # (mas experiment_conductor.py:430). An interceptor accurate enough to reach a
        # sub-0.2 m CPA therefore flies INTO the target airframe (~0.5 m across). On the
        # RAL 055 E56 DGL cohort that is exactly what happened: trial A-0 hit a 0.104 m
        # CPA, the collision cost the target PX4 its attitude/accel-bias estimate
        # ("Failsafe activated" -> "mc_pos_control: Failsafe: blind land"), and it came to
        # rest INVERTED (roll 178.7 deg) on the ground. Nothing recovers that from inside a
        # running stack: preflight then rejects arming (High Accelerometer Bias, Attitude
        # failure), and the conductor's _reposition FLIES vehicles to the IC rather than
        # teleporting, so a downed vehicle can never settle. Every later trial then
        # measured range-to-corpse (121.88 m) and both boots aborted 1-valid-of-16.
        # Uncensored CPA and physical contact are the same property; this filter keeps the
        # metric and drops the contact.
        #
        # DEFAULT ON as of 2026-08-08 (user decision: the remaining cohort runs filtered,
        # and an opt-IN is something an operator can forget). Until then the default was
        # OFF and the filter was delivered per-boot via `tmux setenv -g`; no tmuxp file
        # ever set it, so a forgotten setenv silently bought back the E56 failure mode.
        #   DISABLE_VEHICLE_COLLISION unset or =1  -> px4_* do not collide with px4_* (default)
        #   DISABLE_VEHICLE_COLLISION=0            -> EXPLICIT OPT-OUT, collision-enabled physics
        # Any other value raises rather than picking a regime silently (as CAMERA_RES does).
        #
        # COMPARABILITY. This flips the physics default, so this launcher is no longer
        # byte-for-byte equivalent to pre-2026-08-08 boots. Every cohort acquired before
        # this commit — RAL 055 E56/E57 and everything earlier — flew collision-ENABLED;
        # E58/E59 flew filtered. Reproducing an older cohort means setting
        # DISABLE_VEHICLE_COLLISION=0 explicitly. See RAL 055 CONSUMER_MANIFEST.md §5.
        #
        # Whichever way it is set, deliver it via the tmuxp `environment:` block or
        # `tmux setenv -g` and read back with `tmux show-environment -g`: `VAR=... tmuxp
        # load` is silently dropped whenever a tmux server exists (same trap as
        # CAMERA_VEHICLES above). That trap is why the opt-out must not be the quiet path.
        # Empty counts as unset (as CAMERA_VEHICLES treats it): the tmuxp files pass this
        # inline as `DISABLE_VEHICLE_COLLISION=${DISABLE_VEHICLE_COLLISION}`, which expands
        # to empty if the `environment:` block ever fails to reach the pane. Falling to the
        # safe default beats aborting the boot, and write_boot_label records the raw value
        # either way, so an empty expansion can never mis-label a cohort.
        _nocollide_raw = os.environ.get("DISABLE_VEHICLE_COLLISION", "").strip()
        _nocollide_env = _nocollide_raw or "1"
        if _nocollide_env not in ("0", "1"):
            raise ValueError(
                "unknown DISABLE_VEHICLE_COLLISION=%r (use 0|1; unset or empty means 1, the "
                "default since 2026-08-08) — refusing to guess a collision regime"
                % _nocollide_raw)
        if _nocollide_env == "1":
            _collision_group = self.disable_vehicle_vehicle_collision(num_vehicles)
        else:
            _collision_group = None
            print("[PegasusApp] *** DISABLE_VEHICLE_COLLISION=0 — EXPLICIT OPT-OUT: "
                  "vehicle-vehicle collision is ON (pre-2026-08-08 physics). An interceptor "
                  "reaching a sub-0.5 m CPA can down a vehicle and invalidate the rest of "
                  "the cohort — see RAL ticket 055 E56. ***")

        # The machine-readable half of the boot witness (RAL ticket 055 §4 follow-up).
        self.write_boot_label(num_vehicles, _nocollide_raw, _collision_group)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Apply camera prim properties after world.reset() (render products exist now)
        # focus_distance and clipping_range match iris_ma6 training env (iris_ma_env6_test_cfg.py).
        f_stop = 1.8
        focus_distance = 400.0  # matches training PinholeCameraCfg.focus_distance
        for vehicle in self.vehicles:
            for sensor in vehicle._graphical_sensors:
                sensor._camera.set_focal_length(self._cam_focal_length / 10.0)     # mm → cm
                sensor._camera.set_focus_distance(focus_distance)
                sensor._camera.set_lens_aperture(f_stop * 100.0)
                sensor._camera.set_horizontal_aperture(self._cam_h_aperture / 10.0) # mm → cm
                sensor._camera.set_vertical_aperture(self._cam_v_aperture / 10.0)
                sensor._camera.set_clipping_range(0.1, 1.0e5)

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def disable_vehicle_vehicle_collision(self, num_vehicles: int):
        """Author a SELF-FILTERING UsdPhysics.CollisionGroup over the vehicle prims.

        A collision group that lists ITSELF in `filteredGroups` means its members do not
        collide with one another, while collisions with everything OUTSIDE the group —
        notably the ground plane — are untouched. That is exactly the semantics RAL
        ticket 055 S21 needs: an interceptor may fly through the target to its true CPA,
        but a vehicle that falls still lands.

        Fail-closed by construction. A filter that silently fails to author is worse than
        no filter at all, because the cohort would look protected, run for 40 minutes and
        then lose a vehicle anyway — which is the failure this exists to prevent. So the
        prims are checked before authoring and the authored targets are read back after.

        Returns the READBACK-VERIFIED group description (path, filteredGroups, members) so
        the boot label reports what was authored rather than what was requested.
        """
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        group_path = "/World/CollisionGroups/vehicles"
        group = UsdPhysics.CollisionGroup.Define(stage, group_path)

        # Self-filtering: members of this group ignore each other.
        filtered = group.CreateFilteredGroupsRel()
        filtered.AddTarget(group_path)

        # Membership. Newer USD exposes the collection through the schema; fall back to
        # the raw Usd.CollectionAPI instance name ("colliders") on older builds.
        try:
            includes = group.GetCollidersCollectionAPI().CreateIncludesRel()
        except Exception:
            includes = Usd.CollectionAPI.Apply(
                group.GetPrim(), "colliders").CreateIncludesRel()

        members = []
        for i in range(1, num_vehicles + 1):
            prim_path = "/World/" + self.namespace + str(i)
            if not stage.GetPrimAtPath(prim_path).IsValid():
                raise RuntimeError(
                    "DISABLE_VEHICLE_COLLISION: vehicle prim %s is missing — refusing to "
                    "boot with a filter that would silently not apply" % prim_path)
            includes.AddTarget(prim_path)
            members.append(prim_path)

        authored_filter = [t.pathString for t in filtered.GetTargets()]
        authored_members = [t.pathString for t in includes.GetTargets()]
        if group_path not in authored_filter or sorted(authored_members) != sorted(members):
            raise RuntimeError(
                "DISABLE_VEHICLE_COLLISION: readback mismatch — filteredGroups=%s "
                "includes=%s (expected self-filter on %s over %s)"
                % (authored_filter, authored_members, group_path, members))

        print("[PegasusApp] DISABLE_VEHICLE_COLLISION=1 -> self-filtering CollisionGroup "
              "%s over %d vehicles %s (vehicle-GROUND collision UNCHANGED) [RAL ticket 055]"
              % (group_path, len(members), members))

        return {"path": group_path,
                "filtered_groups": authored_filter,
                "members": authored_members,
                "ground_collision": "unchanged"}

    def write_boot_label(self, num_vehicles: int, nocollide_raw: str, collision_group):
        """Write the Isaac-side boot label — the machine-readable half of the boot witness.

        WHY. RAL ticket 055 `s21_deviation_deploy_assertion.md` §4: the collision regime,
        the single variable separating cohorts E58/E59 from E57/E56, is absent from the
        conductor's provenance entirely — "a reader of the provenance JSONs alone cannot
        distinguish E58 from E57" — because `experiment_conductor` has no visibility into
        THIS process's environment. Only Isaac knows the Isaac-side configuration, so
        Isaac writes it down. That ticket's filed follow-up is the conductor archiving
        this file into `boot_<id>_provenance.json`.

        It labels what was AUTHORED, not what was requested: `collision_group` is the
        readback-verified description returned by disable_vehicle_vehicle_collision(), so
        a filter that failed to apply cannot produce a label claiming it did.

        STALENESS is the hazard a well-known path invites — a consumer that reads a label
        left behind by a previous session would silently mis-label a whole cohort, which
        is the exact failure class this file exists to close. So the label carries this
        process's `pid` and `boot_unix_time`, and a consumer MUST check that `/proc/<pid>`
        is alive before trusting it (the conductor runs on this same box). The file is
        written atomically, so a reader never sees a half-written label.

        Fail-closed, like the filter above: if the label cannot be written the boot stops.
        A cohort that cannot state its own physics is the thing being prevented.
        """
        import hashlib
        import json
        import socket
        import time

        producer = os.path.abspath(__file__)
        try:
            with open(producer, "rb") as f:
                producer_sha256 = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            producer_sha256 = None

        label = {
            "schema": "isaac_boot_label/1",
            # Liveness — a consumer must verify /proc/<pid> before trusting this file.
            "pid": os.getpid(),
            "boot_unix_time": time.time(),
            "hostname": socket.gethostname(),
            "producer": producer,
            "producer_sha256": producer_sha256,
            "headless": bool(args.headless),
            # The Isaac-side configuration surface that is invisible downstream.
            "num_vehicles": num_vehicles,
            "namespace": self.namespace,
            "camera_res": self._camera_res,
            "camera_width": self._cam_width,
            "camera_height": self._cam_height,
            "camera_vehicles": sorted(self._camera_ids),
            "disable_vehicle_collision": collision_group is not None,
            # The raw env string as delivered ("" = unset, i.e. the default). The boolean
            # above is derived from the AUTHORED group and is the authoritative field.
            "disable_vehicle_collision_env": nocollide_raw,
            "collision_group": collision_group,
        }

        path = os.environ.get("ISAAC_BOOT_LABEL", "/tmp/isaac_boot_label.json").strip()
        tmp = path + ".%d.tmp" % os.getpid()
        try:
            with open(tmp, "w") as f:
                json.dump(label, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)          # atomic: no reader sees a partial label
        except OSError as e:
            raise RuntimeError(
                "ISAAC_BOOT_LABEL: could not write the boot label to %s (%s) — refusing to "
                "boot a cohort that cannot state its own Isaac-side configuration "
                "[RAL ticket 055]" % (path, e))

        print("[PegasusApp] ISAAC_BOOT_LABEL -> %s (pid=%d, disable_vehicle_collision=%s) "
              "[RAL ticket 055]" % (path, label["pid"], label["disable_vehicle_collision"]))

    def load_flight_scene(self, usd_path: str, scale: float = 0.001, offset: tuple = (0.0, 0.0, 0.0)):
        """Load the Flight aesthetic scene USD with Y-up → Z-up rotation.

        Args:
            usd_path: Absolute path to the Flight USD file.
            scale: Uniform scale (USD is in cm with large coordinates; 0.001 fits the env).
            offset: (x, y, z) translation offset in meters after scaling.
        """
        import omni.usd
        from pxr import UsdGeom, Gf

        stage = omni.usd.get_context().get_stage()
        xform = UsdGeom.Xform.Define(stage, "/World/FlightScene")
        # Order: translate → rotate → scale (USD applies ops top-to-bottom)
        ox, oy, oz = offset
        xform.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
        xform.AddRotateXOp().Set(90.0)  # Y-up → Z-up
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        xform.GetPrim().GetReferences().AddReference(usd_path)

    def create_landmarks(self):
        from omni.isaac.core.objects import DynamicCuboid
        import numpy as np
        cube_1 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/new_cube_1",
                name="cube_1",
                position=np.array([self.world_offset_x + 8.0, self.world_offset_y + 0, self.world_offset_z + 1.0]),
                scale=np.array([1.0, 1.0, 1.0]),
                size=1.0,
                color=np.array([255, 0, 0]),
            )
        )
        cube_2 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/new_cube_2",
                name="cube_2",
                position=np.array([self.world_offset_x + 0.0, self.world_offset_y + 8.0, self.world_offset_z + 1.0]),
                scale=np.array([1.0, 1.0, 1.0]),
                size=1.0,
                color=np.array([0, 255, 0]),
            )
        )
        cube_3 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/new_cube_3",
                name="cube_3",
                position=np.array([self.world_offset_x + 0, self.world_offset_y + 0, self.world_offset_z + 1.0]),
                scale=np.array([1.0, 1.0, 1.0]),
                size=1.0,
                color=np.array([0, 0, 255]),
            )
        )

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        """Auxiliar method to create multiple multirotor vehicles

        Args:
            vehicle_id (_type_): _description_
        """

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # RAL ticket 019 revisit2 W0f: does this vehicle carry a camera at all?
        _has_camera = vehicle_id in self._camera_ids

        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [
            PX4MavlinkBackend(mavlink_config),
            ROS2Backend(vehicle_id=vehicle_id,
                        config={
                            "namespace": self.namespace,
                            "pub_sensors": True,
                            "pub_graphical_sensors": _has_camera,
                            "pub_state": True,
                            "sub_control": False,
                            "sub_zoom": True,
                        })]

        # RAL ticket 019 revisit2 W0f: skip the render product entirely for vehicles that have
        # no camera consumer (see CAMERA_VEHICLES above). An empty graphical_sensors list is
        # safe by construction — the post-world.reset() property loop iterates
        # `vehicle._graphical_sensors` and simply does nothing — and the USD prim is untouched.
        if _has_camera:
            config_multirotor.graphical_sensors = [
                MonocularCamera("/pitch_link/camera",
                config={
                    "frequency": 25,
                    "resolution": (self._cam_width, self._cam_height),
                    "position": np.array([0, 0, 0]),
                    "orientation": np.array([0.0, 0.0, 0.0]),
                    "intrinsics": np.array(self._cam_matrix),
                    }
                )
            ]
        else:
            config_multirotor.graphical_sensors = []
            print(f"[PegasusApp] vehicle {vehicle_id}: camera SKIPPED (not in CAMERA_VEHICLES)")

        vehicle_name = self.namespace + str(vehicle_id)
        vehicle_stage_path = "/World/" + vehicle_name
        # vehicle_stage_path = "/World/quadrotor"


        self.vehicles += [Multirotor(
            vehicle_stage_path,
            # ROBOTS['Iris'],
            ROBOTS['IrisGimbal3'],
            vehicle_id,
            [self.world_offset_x + gap_x_axis * (vehicle_id+1), self.world_offset_y + gap_x_axis * (vehicle_id+1), self.world_offset_z + 2.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 3.14], degrees=True).as_quat(),
            config=config_multirotor)]
        asyncio.ensure_future(self.create_ros_action_graph(vehicle_stage_path, vehicle_name))
        # asyncio.ensure_future(self.create_ros_camera_graph(vehicle_stage_path, vehicle_name))

    async def create_ros_camera_graph(self, vehicle_stage_path, vehicle_name):
        try:
            await omni.kit.app.get_app().next_update_async()
            camera_graph = bridge.Ros2CameraGraph()
            camera_graph._og_path = vehicle_stage_path + "/CameraGraph"
            camera_graph._camera_prim = vehicle_name + "/pitch_link/camera"
            camera_graph._node_namespace = vehicle_name
            camera_graph.make_graph()
        except Exception as e:
            print(e)
        pass

    async def create_simulation_time_graph(self):
        try:
            await omni.kit.app.get_app().next_update_async()
            og.Controller.edit(
                {"graph_path": "/World/SimulationTimeGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("Context.outputs:context", "PublishClock.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:topicName", "/clock"),
                    ],
                },
            )
            await omni.kit.app.get_app().next_update_async()
        except Exception as e:
            print(e)
        pass


    async def create_ros_action_graph(self, vehicle_stage_path, vehicle_name):
        try:
            await omni.kit.app.get_app().next_update_async()
            og.Controller.edit(
                {"graph_path": vehicle_stage_path + "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                        ("Context.outputs:context", "PublishJointState.inputs:context"),
                        ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                        ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                        (
                            "SubscribeJointState.outputs:positionCommand",
                            "ArticulationController.inputs:positionCommand",
                        ),
                        (
                            "SubscribeJointState.outputs:velocityCommand",
                            "ArticulationController.inputs:velocityCommand",
                        ),
                        ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        # Setting the /Franka target prim to Articulation Controller node
                        ("ArticulationController.inputs:robotPath", vehicle_stage_path + "/body"),
                        ("PublishJointState.inputs:topicName", vehicle_name + "/isaac_joint_states"),
                        ("SubscribeJointState.inputs:topicName", vehicle_name + "/isaac_joint_commands"),
                        ("PublishJointState.inputs:targetPrim", [vehicle_stage_path + "/body"]),
                    ],
                },
            )
            await omni.kit.app.get_app().next_update_async()
        except Exception as e:
            print(e)
        pass

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
