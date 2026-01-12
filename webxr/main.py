import argparse
import os
import numpy as np
from isaacsim import SimulationApp

# 시뮬레이션 앱 실행
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from omni.isaac.core.objects import VisualSphere 
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim, is_prim_path_valid
from omni.isaac.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

# mainHandTeleop.py 파일에 위에서 작성한 LeapVectorMapping 클래스가 있다고 가정합니다.
from mainHandTeleop import TesolloVectorMapping

class loadUSD():
    def __init__(self):
        # 경로 설정
        self.base_path = "/home/youngwoo/TeleopTesollo/assets/new_rmpflow"
        self.urdf_path = "/home/youngwoo/openarm_tesollo_urdf/openarm_tesollo.urdf"
        
        # Left Arm Config
        self.left_robot_yaml = os.path.join(self.base_path, "openarm_robot_left.yaml")
        self.left_rmp_config = os.path.join(self.base_path, "openarm_rmpflow_common.yaml")
        
        # Right Arm Config
        self.right_robot_yaml = os.path.join(self.base_path, "openarm_robot_right.yaml")
        self.right_rmp_config = os.path.join(self.base_path, "openarm_rmpflow_common.yaml")

        self.calibrated = False
        self.left_hand_offset_pos = None
        self.right_hand_offset_pos = None

        # RMPflow 객체들
        self.left_rmpflow = None
        self.right_rmpflow = None
        self.left_policy = None
        self.right_policy = None
        
        # 시각화용 구체 객체 저장 변수
        self.left_vis_sphere = None
        self.right_vis_sphere = None
        
        # 관절 인덱스 저장용
        self.left_active_indices = []
        self.right_active_indices = []

        self.local_left_start_pos = np.array([-0.35, -0.25, -0.4]) # 왼손이 Isaac Sim에서 시작 초기에 어디에 위치될건지
        self.local_right_start_pos = np.array([-0.35, 0.25, -0.4]) # 오른손이 Isaac Sim에서 시작 초기에 어디에 위치될건지


    def setup_scene(self, world: World, ROBOT_PATH, ROBOT_PRIM_PATH) -> SingleArticulation:
        world.get_physics_context().set_gravity(value=0.0)
        world.scene.add_default_ground_plane(prim_path=f"{ROBOT_PRIM_PATH}")
        if not is_prim_path_valid("/World/DomeLight"):
            define_prim("/World/DomeLight", "DomeLight")
            
        add_reference_to_stage(usd_path=ROBOT_PATH, prim_path=ROBOT_PRIM_PATH)
        
        robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="Robot")
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.0]))
        world.scene.add(robot)
        
        # 디버깅용 Sphere 생성
        self.left_vis_sphere = VisualSphere(
            prim_path="/World/LeftTargetVis", 
            name="left_target_vis",
            radius=0.03,
            color=np.array([1.0, 0.0, 0.0]) # 왼손 : 빨간색 공으로 표현
        )
        self.left_vis_sphere.set_collision_enabled(False)

        self.right_vis_sphere = VisualSphere(
            prim_path="/World/RightTargetVis", 
            name="right_target_vis",
            radius=0.03,
            color=np.array([0.0, 0.0, 1.0]) # 오른손 : 파란색 공으로 표현
        )
        self.right_vis_sphere.set_collision_enabled(False)

        
        # Left RmpFlow 설정
        left_config = {
            "robot_description_path": self.left_robot_yaml,
            "urdf_path": self.urdf_path,
            "rmpflow_config_path": self.left_rmp_config,
            "end_effector_frame_name": "openarm_left_link7", # left_link6 -> openarm_left_link7
            "maximum_substep_size": 0.00334
        }
        self.left_rmpflow = RmpFlow(**left_config)
        self.left_policy = ArticulationMotionPolicy(robot, self.left_rmpflow)
        
        # Right RmpFlow 설정
        right_config = {
            "robot_description_path": self.right_robot_yaml,
            "urdf_path": self.urdf_path,
            "rmpflow_config_path": self.right_rmp_config,
            "end_effector_frame_name": "openarm_right_link7", # right_hand_arm_link6 -> openarm_right_link7
            "maximum_substep_size": 0.00334
        }
        self.right_rmpflow = RmpFlow(**right_config)
        self.right_policy = ArticulationMotionPolicy(robot, self.right_rmpflow)
        
        print("RmpFlow Initialized with openarm_link7 frames.")
        return robot
    
    def post_reset_initialize(self, robot):
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.0]))
        all_joint_names = robot.dof_names
        
        left_active_names = self.left_rmpflow.get_active_joints()
        self.left_active_indices = [all_joint_names.index(jn) for jn in left_active_names]
        
        right_active_names = self.right_rmpflow.get_active_joints()
        self.right_active_indices = [all_joint_names.index(jn) for jn in right_active_names]
        
        kps = np.full(robot.num_dof, 1800.0)
        kds = np.full(robot.num_dof, 180.0)
        robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
        print("Robot Gains Force Set.")

    def hand_arm_Teleop(self, lvm, robot: SingleArticulation, left_joints, right_joints):
        dof_names = robot.dof_names
        l_hand_transforms, r_hand_transforms, l_wrist_mat, r_wrist_mat = lvm.get_avp_data_sync()
        
        l_pos, l_rot = None, None
        r_pos, r_rot = None, None

        # 2. Wrist(Arm) 데이터 처리: 이미 행렬이므로 바로 포즈 추출
        if l_wrist_mat is not None:
            l_pos, l_rot = lvm.convert_avp_to_isaac_pose(l_wrist_mat, 'left')

        if r_wrist_mat is not None:
            r_pos, r_rot = lvm.convert_avp_to_isaac_pose(r_wrist_mat, 'right')
            
        # 캘리브레이션 (0점 조절)
        if not self.calibrated:
            if l_pos is not None or r_pos is not None:
                print("Calibrating Arms (Setting Zero Point)...")
                self.left_hand_offset_pos = l_pos if l_pos is not None else np.array([0,0,0])
                self.right_hand_offset_pos = r_pos if r_pos is not None else np.array([0,0,0])
                self.calibrated = True
            return None 

        full_target_positions = robot.get_joint_positions()
        IK_scaler = 1.0
        dt = 1.0

        # 로봇의 현재 월드 위치/회전
        base_pos, base_rot = robot.get_world_pose()
        

        # --- Left Arm & Hand Control ---
        if l_pos is not None:
            # 1. 오프셋 계산 (이미 Isaac 좌표계임)
            delta_pos = (l_pos - self.left_hand_offset_pos) * IK_scaler
            
            # 2. 타겟 위치 설정 (수동 축 교체 삭제)
            target_pos = base_pos + self.local_left_start_pos + delta_pos
            
            # targetX = target_pos[0]
            # targetY = -target_pos[2]
            # targetZ = target_pos[1]
            # calib_target_pos = np.array([targetX, targetY, targetZ])
                        # 3. 시각화와 로봇 타겟을 동일하게 설정

            # 추가적인 * -1 연산 없이 그대로 전달            
            armtargetX = -target_pos[0]
            armtargetY = -target_pos[1]
            armtargetZ = -target_pos[2]

            arm_calib_target_pos = np.array([armtargetX, armtargetY, armtargetZ])
            if self.left_vis_sphere:
                self.left_vis_sphere.set_world_pose(position=arm_calib_target_pos)
            self.left_rmpflow.set_end_effector_target(arm_calib_target_pos, l_rot)
            self.left_rmpflow.update_world()
            
            left_action = self.left_policy.get_next_articulation_action(dt)
            if left_action.joint_positions is not None:
                if left_action.joint_indices is not None:
                     full_target_positions[left_action.joint_indices] = left_action.joint_positions
                else:
                     full_target_positions[self.left_active_indices] = left_action.joint_positions
            
            # Hand Control
            if l_hand_transforms is not None and len(l_hand_transforms) > 0:
                left_hand_degrees = lvm.calculate_hand_angles(l_hand_transforms)
                left_finger_action_full = lvm.map_to_left_robot_action(left_hand_degrees, dof_names, left_joints)
                
                nonzero_indices = np.nonzero(left_finger_action_full)[0]
                if len(nonzero_indices) > 0:
                    full_target_positions[nonzero_indices] = left_finger_action_full[nonzero_indices]

        # --- Right Arm & Hand Control ---
        if r_pos is not None:
            # Arm Control
            delta_pos = (r_pos - self.right_hand_offset_pos) * IK_scaler
            target_pos = base_pos + self.local_right_start_pos + delta_pos
            
        
            # targetX = target_pos[0]
            # targetY = -target_pos[2]
            # targetZ = target_pos[1]
            # calib_target_pos = np.array([targetX, targetY, targetZ])
            armtargetX = -target_pos[0]
            armtargetY = -target_pos[1]
            armtargetZ = -target_pos[2]
            arm_calib_target_pos = np.array([armtargetX, armtargetY, armtargetZ])
            if self.right_vis_sphere:
                self.right_vis_sphere.set_world_pose(position=arm_calib_target_pos)

            self.right_rmpflow.set_end_effector_target(arm_calib_target_pos, r_rot)
            self.right_rmpflow.update_world()
            
            right_action = self.right_policy.get_next_articulation_action(dt)
            if right_action.joint_positions is not None:
                if right_action.joint_indices is not None:
                    full_target_positions[right_action.joint_indices] = right_action.joint_positions
                else:
                    full_target_positions[self.right_active_indices] = right_action.joint_positions
            
            # [수정] Hand Control (Fingers)
            if r_hand_transforms is not None and len(r_hand_transforms) > 0:
                right_hand_degrees = lvm.calculate_hand_angles(r_hand_transforms)
                right_finger_action_full = lvm.map_to_right_robot_action(right_hand_degrees, dof_names, right_joints)
                
                nonzero_indices = np.nonzero(right_finger_action_full)[0]
                if len(nonzero_indices) > 0:
                    full_target_positions[nonzero_indices] = right_finger_action_full[nonzero_indices]

        return ArticulationAction(joint_positions=full_target_positions)

    def run_simulator(self, world: World, robot: SingleArticulation):
        print(f"Robot DoF Names: {robot.dof_names}")
        self.post_reset_initialize(robot)
        lvm = TesolloVectorMapping()
        left_joints, right_joints = lvm.get_joint_index(robot.dof_names)
        print(left_joints, right_joints)
        while simulation_app.is_running():
            if world.is_playing():
                world.step(render=False) 
                
                action = self.hand_arm_Teleop(lvm, robot, left_joints, right_joints)
                
                if action is not None:
                    robot.apply_action(action)
            
            world.step(render=True) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()
    
    world = World(stage_units_in_meters=1.0)
    loading = loadUSD()
    
    usdPath = "/home/youngwoo/openarm_tesollo_urdf/openarm_tesollo.usd"
    primPath = "/World/openarm_tesollo_mount"
    
    robot = loading.setup_scene(world, usdPath, primPath)
    
    world.reset()
    world.step(render=True) 
    
    # set_camera_view([0.0, -2.0, 2.0], [0.0, 0.0, 0.5])
    
    loading.run_simulator(world, robot)

if __name__ == "__main__":
    main()