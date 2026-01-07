import argparse
import os
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import numpy as np
import time
from isaacsim.core.api import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim, is_prim_path_valid
from omni.isaac.core.utils.viewports import set_camera_view
from mainHandTeleop import LeapVectorMapping

class loadUSD():
    def __init__(self):
        self.left_calibration_offset = None
        self.right_calibration_offset = None
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.left_wrist_pos = None
        self.right_wrist_pos = None
        self.left_wrist_roll = None
        self.right_wrist_roll = None

    def setup_scene(self, world: World, ROBOT_PATH, ROBOT_PRIM_PATH) -> Articulation:
        world.get_physics_context().set_gravity(value=0.0)
        world.scene.add_default_ground_plane(prim_path=f"{ROBOT_PRIM_PATH}")
        if not is_prim_path_valid("/World/DomeLight"):
            define_prim("/World/DomeLight", "DomeLight")
            
        add_reference_to_stage(usd_path=ROBOT_PATH, prim_path=ROBOT_PRIM_PATH)
        robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="Robot", position=np.array([0.0, 0.0, 1.0]))
        world.scene.add(robot)
        return robot
    
    def handTeleop(self, pbik, dof_names):
        total_action = np.zeros(len(dof_names))
        self.left_hand_pos, self.right_hand_pos = pbik.get_avp_data()
        #왼손
        if self.left_hand_pos is not None:
            left_hand_degrees = pbik.calculate_hand_angles(self.left_hand_pos)
            left_current_action = pbik.map_to_left_robot_action(left_hand_degrees, dof_names)
            if self.left_calibration_offset is None:
                print('Calibration... Keep left hand open!')
                self.left_calibration_offset = left_current_action
                left_target_action = np.zeros_like(left_current_action)
            else:
                left_target_action = left_current_action - self.left_calibration_offset
                left_target_action = np.clip(left_target_action, -2.0, 2.0)
            total_action += left_target_action
        #오른손
        if self.right_hand_pos is not None:
            right_hand_degrees = pbik.calculate_hand_angles(self.right_hand_pos)
            right_current_action = pbik.map_to_right_robot_action(right_hand_degrees, dof_names)
            if self.right_calibration_offset is None:
                print('Calibrating... Keep right hand open!')
                self.right_calibration_offset = right_current_action
                right_target_action = np.zeros_like(right_current_action)
            else:
                right_target_action = right_current_action - self.right_calibration_offset
                right_target_action = np.clip(right_target_action, -2.0, 2.0)
            total_action += right_target_action

        return total_action


    def run_simulator(self, world: World, robot: Articulation):
        pbik = LeapVectorMapping()
        dof_names = robot.dof_names
        print(f"Robot DoF Names: {dof_names}")
        
        while simulation_app.is_running():
            hand_target_action = self.handTeleop(pbik, dof_names)
            if hand_target_action is not None:
                robot.apply_action(ArticulationAction(joint_positions=hand_target_action))

            world.step(render=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()
    world = World(stage_units_in_meters=1.0)
    loading = loadUSD()
    usdPath = "/home/youngwoo/newTeleop/assets/full_robot/full_robot.usd"
    primPath = "/World/full_robot"
    robot = loading.setup_scene(world, usdPath, primPath)
    world.reset()
    world.render()
    set_camera_view([0.0, -2.0, 2.0], [0.0, 0.0, 0.5])
    
    loading.run_simulator(world, robot)

if __name__ == "__main__":
    main()