import argparse
import os
import numpy as np
from isaacsim import SimulationApp

# 시뮬레이션 앱 실행
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
# [추가] 시각화용 구체(Sphere) import
from omni.isaac.core.objects import VisualSphere 
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim, is_prim_path_valid
from omni.isaac.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

from mainHandTeleop import LeapVectorMapping

class loadUSD():
    def __init__(self):
        # 경로 설정
        self.base_path = "/home/youngwoo/newTeleop/rmpflow"
        self.urdf_path = os.path.join(self.base_path, "robot/urdf/dual_arm_robot.urdf")
        
        # Left Arm Config
        self.left_robot_yaml = os.path.join(self.base_path, "robot/urdf/dual_arm_robot_left.yaml")
        self.left_rmp_config = os.path.join(self.base_path, "dual_left_rmpflow_common.yaml")
        
        # Right Arm Config
        self.right_robot_yaml = os.path.join(self.base_path, "robot/urdf/dual_arm_robot_right.yaml")
        self.right_rmp_config = os.path.join(self.base_path, "dual_right_rmpflow_common.yaml")

        self.calibrated = False
        self.left_hand_offset_pos = None
        self.right_hand_offset_pos = None

        # RMPflow 객체들
        self.left_rmpflow = None
        self.right_rmpflow = None
        self.left_policy = None
        self.right_policy = None
        
        # [추가] 시각화용 구체 객체 저장 변수
        self.left_vis_sphere = None
        self.right_vis_sphere = None
        
        # 관절 인덱스 저장용
        self.left_active_indices = []
        self.right_active_indices = []

        self.local_left_start_pos = np.array([-0.3, 0.5, 0.5]) 
        self.local_right_start_pos = np.array([0.3, 0.5, 0.5]) 


    def setup_scene(self, world: World, ROBOT_PATH, ROBOT_PRIM_PATH) -> SingleArticulation:
        world.get_physics_context().set_gravity(value=0.0)
        world.scene.add_default_ground_plane(prim_path=f"{ROBOT_PRIM_PATH}")
        if not is_prim_path_valid("/World/DomeLight"):
            define_prim("/World/DomeLight", "DomeLight")
            
        add_reference_to_stage(usd_path=ROBOT_PATH, prim_path=ROBOT_PRIM_PATH)
        
        robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="Robot")
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.7]))
        world.scene.add(robot)
        
        # [추가] 디버깅용 Sphere 생성 (빨강: 왼쪽 / 파랑: 오른쪽)
        # 반지름 0.03 (3cm)
        self.left_vis_sphere = VisualSphere(
            prim_path="/World/LeftTargetVis", 
            name="left_target_vis",
            radius=0.03,
            color=np.array([1.0, 0.0, 0.0]) # Red
        )
        self.left_vis_sphere.set_collision_enabled(False) # 로봇이랑 충돌 안하게 설정

        self.right_vis_sphere = VisualSphere(
            prim_path="/World/RightTargetVis", 
            name="right_target_vis",
            radius=0.03,
            color=np.array([0.0, 0.0, 1.0]) # Blue
        )
        self.right_vis_sphere.set_collision_enabled(False)

        
        # Left RmpFlow 설정
        left_config = {
            "robot_description_path": self.left_robot_yaml,
            "urdf_path": self.urdf_path,
            "rmpflow_config_path": self.left_rmp_config,
            "end_effector_frame_name": "left_link6", 
            "maximum_substep_size": 0.00334
        }
        self.left_rmpflow = RmpFlow(**left_config)
        self.left_policy = ArticulationMotionPolicy(robot, self.left_rmpflow)
        
        # Right RmpFlow 설정
        right_config = {
            "robot_description_path": self.right_robot_yaml,
            "urdf_path": self.urdf_path,
            "rmpflow_config_path": self.right_rmp_config,
            "end_effector_frame_name": "right_hand_arm_link6", 
            "maximum_substep_size": 0.00334
        }
        self.right_rmpflow = RmpFlow(**right_config)
        self.right_policy = ArticulationMotionPolicy(robot, self.right_rmpflow)
        
        print("RmpFlow Initialized Successfully.")
        return robot
    
    def post_reset_initialize(self, robot):
        # Active Joint 인덱스 찾기
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.0]))
        all_joint_names = robot.dof_names
        
        left_active_names = self.left_rmpflow.get_active_joints()
        self.left_active_indices = [all_joint_names.index(jn) for jn in left_active_names]
        
        right_active_names = self.right_rmpflow.get_active_joints()
        self.right_active_indices = [all_joint_names.index(jn) for jn in right_active_names]
        
        kps = np.full(robot.num_dof, 40000.0)
        kds = np.full(robot.num_dof, 1000.0)
        robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
        print("Robot Gains Force Set.")

    def hand_arm_Teleop(self, pbik, robot: SingleArticulation, left_joints, right_joints):
        dof_names = robot.dof_names
        
        # 데이터 수신
        left_hand_fingers, right_hand_fingers, left_wrist_mat, right_wrist_mat = pbik.get_avp_data()
        
        # [중요] 회전 문제 디버깅을 위해 일단 Rotation은 무시하거나 고정된 값 사용 추천
        # pbik.convert_avp_to_isaac_pose 내부에서 Identity Quaternion 리턴하도록 해두셨다면 그대로 사용
        l_pos, l_rot = pbik.convert_avp_to_isaac_pose(left_wrist_mat, 'left')
        r_pos, r_rot = pbik.convert_avp_to_isaac_pose(right_wrist_mat, 'right')
        # if l_rot is not None:
        #     l_rot = l_rot * np.array([1, -1, -1, -1])
            
        # if r_rot is not None:
        #     r_rot = r_rot * np.array([1, -1, -1, -1])
            
        if not self.calibrated:
            if l_pos is not None or r_pos is not None:
                print("Calibrating Arms (Setting Zero Point)...")
                # 현재 내 손의 위치를 '0점'으로 잡음
                self.left_hand_offset_pos = l_pos if l_pos is not None else np.array([0,0,0])
                self.right_hand_offset_pos = r_pos if r_pos is not None else np.array([0,0,0])
                self.calibrated = True
            return None 

        full_target_positions = robot.get_joint_positions()
        IK_scaler = 1.0
        dt = 1.0

        # [핵심 추가] 로봇의 현재 월드 위치와 회전을 가져옵니다.
        # 로봇이 움직이면 base_pos 값이 계속 바뀝니다.
        base_pos, base_rot = robot.get_world_pose()

        # --- Left Arm Control ---
        if l_pos is not None:
            # 1. 내 손의 이동량 계산 (Delta)
            delta_pos = (l_pos - self.left_hand_offset_pos) * IK_scaler
            
            # 2. [수정] 최종 타겟 = 로봇 현재 위치 + 로컬 오프셋 + 내 손 이동량
            # 이렇게 하면 로봇이 어디에 있든 "로봇 기준"으로 움직입니다.
            target_pos = base_pos + self.local_left_start_pos + delta_pos
            
            # (심화) 만약 로봇이 '회전'까지 한다면 delta_pos에도 base_rot을 곱해줘야 하지만,
            # 일단 위치 이동만 한다면 위 식만으로 충분합니다.

            # 시각화
            if self.left_vis_sphere:
                self.left_vis_sphere.set_world_pose(position=target_pos)
            nexXtarget = target_pos[0] * -1
            newYtarget = target_pos[1] * -1
            newZtarget = target_pos[2] * 1
            arm_target_pos = np.array([nexXtarget, newYtarget, newZtarget])
            
            self.left_rmpflow.set_end_effector_target(arm_target_pos, l_rot)
            self.left_rmpflow.update_world()
            
            left_action = self.left_policy.get_next_articulation_action(dt)
            if left_action.joint_positions is not None:
                if left_action.joint_indices is not None:
                     full_target_positions[left_action.joint_indices] = left_action.joint_positions
                else:
                     full_target_positions[self.left_active_indices] = left_action.joint_positions
            
            # 손가락 매핑
            left_hand_degrees = pbik.calculate_hand_angles(left_hand_fingers)
            left_finger_action_full = pbik.map_to_left_robot_action(left_hand_degrees, dof_names, left_joints)
            nonzero_indices = np.nonzero(left_finger_action_full)[0]
            if len(nonzero_indices) > 0:
                full_target_positions[nonzero_indices] = left_finger_action_full[nonzero_indices]

        # --- Right Arm Control ---
        if r_pos is not None:
            delta_pos = (r_pos - self.right_hand_offset_pos) * IK_scaler
            
            # [수정] 오른쪽도 동일하게 로봇 현재 위치 기준
            target_pos = base_pos + self.local_right_start_pos + delta_pos
            
            if self.right_vis_sphere:
                self.right_vis_sphere.set_world_pose(position=target_pos)
            nexXtarget = target_pos[0] * -1
            newYtarget = target_pos[1] * -1
            newZtarget = target_pos[2] * 1
            arm_target_pos = np.array([nexXtarget, newYtarget, newZtarget])
            self.right_rmpflow.set_end_effector_target(arm_target_pos, r_rot)
            self.right_rmpflow.update_world()
            
            right_action = self.right_policy.get_next_articulation_action(dt)
            if right_action.joint_positions is not None:
                if right_action.joint_indices is not None:
                    full_target_positions[right_action.joint_indices] = right_action.joint_positions
                else:
                    full_target_positions[self.right_active_indices] = right_action.joint_positions
            
            right_hand_degrees = pbik.calculate_hand_angles(right_hand_fingers)
            right_finger_action_full = pbik.map_to_right_robot_action(right_hand_degrees, dof_names, right_joints)
            nonzero_indices = np.nonzero(right_finger_action_full)[0]
            if len(nonzero_indices) > 0:
                full_target_positions[nonzero_indices] = right_finger_action_full[nonzero_indices]

        return ArticulationAction(joint_positions=full_target_positions)

    def run_simulator(self, world: World, robot: SingleArticulation):
        print(f"Robot DoF Names: {robot.dof_names}")
        self.post_reset_initialize(robot)
        pbik = LeapVectorMapping()
        left_joints, right_joints = pbik.get_joint_index(robot.dof_names)

        while simulation_app.is_running():
            if world.is_playing():
                world.step(render=False) 
                
                action = self.hand_arm_Teleop(pbik, robot, left_joints, right_joints)
                
                if action is not None:
                    robot.apply_action(action)
            
            world.step(render=True) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()
    
    world = World(stage_units_in_meters=1.0)
    loading = loadUSD()
    
    usdPath = "/home/youngwoo/newTeleop/rmpflow/robot/rotated_dual_arm_robot.usd"
    primPath = "/World/rotated_dual_arm_robot"
    
    robot = loading.setup_scene(world, usdPath, primPath)
    
    world.reset()
    world.step(render=True) 
    
    set_camera_view([0.0, 2.0, 2.0], [0.0, 0.0, 0.5])
    
    loading.run_simulator(world, robot)

if __name__ == "__main__":
    main()