#!/usr/bin/env python
"""
| File: 2_px4_multi_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with multiple vehicles, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
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
import omni.isaac.ros2_bridge as bridge

import omni                                                     # Provides the core omniverse apis
from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with Lidar sensor

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
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Plane with Light"])
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flight"])
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flight Flat"])
        # self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flight with Collision"])
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Full Warehouse"])
        # self.pg.load_asset(SIMULATION_ENVIRONMENTS["Flight Flat"],  "/World/layout")
        self.world_offset_x, self.world_offset_y, self.world_offset_z = 0,0,0
        asyncio.ensure_future(self.create_simulation_time_graph())
        self.create_landmarks()
        
        self.namespace = "/px4_"
        # Spawn 5 vehicles with the PX4 control backend in the simulation, separated by 1.0 m along the x-axis
        num_vehicle = 1
        for i in range(num_vehicle):
            self.vehicle_factory(i+1, gap_x_axis=1.0)
        

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

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
                            "pub_graphical_sensors": True,
                            "pub_state": True,
                            "sub_control": False,})]

        config_multirotor.graphical_sensors = [
            Lidar("/lidar", 
            config={
                "sensor_configuration": "OS1_REV6_32ch10hz1024res",
                "show_render": True,
                }
            )
        ] # Lidar("lidar")

        vehicle_name = self.namespace + str(vehicle_id)
        vehicle_stage_path = "/World" + vehicle_name
        # vehicle_stage_path = "/World/quadrotor"
        

        Multirotor(
            vehicle_stage_path,
            ROBOTS['Iris'],
            # ROBOTS['IrisGimbal'],
            vehicle_id,
            [self.world_offset_x + gap_x_axis * (vehicle_id+1), self.world_offset_y + gap_x_axis * (vehicle_id+1), self.world_offset_z + 2.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 3.14], degrees=True).as_quat(),
            config=config_multirotor)
        asyncio.ensure_future(self.create_ros_action_graph(vehicle_stage_path, vehicle_name))
        # asyncio.ensure_future(self.create_ros_camera_graph(vehicle_stage_path, vehicle_name))
        
    async def create_ros_camera_graph(self, vehicle_stage_path, vehicle_name):
        try:
            await omni.kit.app.get_app().next_update_async()
            camera_graph = bridge.Ros2CameraGraph()
            # camera_graph._og_path = vehicle_stage_path + "/CameraGraph"
            # camera_graph._camera_prim = vehicle_name + "/cgo3_camera_link/camera"
            # camera_graph._node_namespace = vehicle_name
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
                        ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                        # ("RTFPublisher", "omni.isaac.ros2_bridge.ROS2Publisher"),
                        # ("RTF", "omni.isaac.ros2_bridge.IsaacRealTimeFactor"),

                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "ReadSimTime.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "RTFPublisher.inputs:execIn"),
                        ("Context.outputs:context", "PublishClock.inputs:context"),
                        ("Context.outputs:context", "RTFPublisher.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                        # ("RTF.outputs:rtf", "RTFPublisher.inputs:data"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:topicName", "/clock"),
                        # ("RTFPublisher.inputs:topicName", "/realtime_factor"),
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
                        ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                        ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                        ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                        ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                        # ("RTFPublisher", "omni.isaac.ros2_bridge.ROS2Publisher"),
                        # ("RTF", "omni.isaac.ros2_bridge.IsaacRealTimeFactor"),

                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                        ("Context.outputs:context", "PublishJointState.inputs:context"),
                        ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                        ("Context.outputs:context", "PublishClock.inputs:context"),
                        ("Context.outputs:context", "RTFPublisher.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
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
                        # ("RTF.outputs:rtf", "RTFPublisher.inputs:data"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        # Setting the /Franka target prim to Articulation Controller node
                        ("ArticulationController.inputs:robotPath", vehicle_stage_path),
                        ("PublishJointState.inputs:topicName", vehicle_name + "/isaac_joint_states"),
                        ("SubscribeJointState.inputs:topicName", vehicle_name + "/isaac_joint_commands"),
                        ("PublishJointState.inputs:targetPrim", [vehicle_stage_path]),
                        ("PublishClock.inputs:topicName", vehicle_name + "/clock"),
                        # ("RTFPublisher.inputs:topicName", vehicle_name + "/realtime_factor"),
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
