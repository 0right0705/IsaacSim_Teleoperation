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
from omni.isaac.sensor import Camera
import omni.kit.viewport.utility as viewport_utils
import cv2
import base64
import json
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
        from scipy.spatial.transform import Rotation as R
        self.base_view_rot = R.from_euler('y', 50, degrees=True)

        self.calibrated = False
        self.left_hand_offset_pos = None
        self.right_hand_offset_pos = None

        # [추가] 헤드 트래킹 캘리브레이션 변수
        self.head_calibrated = False
        self.head_offset_pos = None
        self.head_offset_quat = None

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


        #camera 추가
        camera_prim_path = f"{ROBOT_PRIM_PATH}/openarm_base/robot_camera"
        
        # 2. 카메라 위치 및 회전 설정
        # 위치: 베이스의 약간 위, 앞쪽 (x=0.2, y=0, z=0.5 등 로봇 크기에 맞춰 조정)
        # 회전: '살짝 아래를 보게' 하기 위해 Pitch 값을 조정 (단위: degree)
        # Isaac Sim 좌표계(공통): X(앞), Y(왼쪽), Z(위) -> 아래를 보려면 Y축 기준으로 회전
        self.xr_origin_path = f"{ROBOT_PRIM_PATH}/openarm_base/XROrigin"
        xr_camera_path = f"{self.xr_origin_path}/XRCamera"
        
        # 2. XROrigin 생성 및 위치 설정
        if not is_prim_path_valid(self.xr_origin_path):
            define_prim(self.xr_origin_path, "XROrigin")
            
            from omni.isaac.core.utils.prims import get_prim_at_path
            from pxr import Gf, UsdGeom  # UsdGeom 임포트 추가
            
            origin_prim = get_prim_at_path(self.xr_origin_path)
            
            # [수정 포인트] 속성을 바로 Set 하는 대신 AddTranslateOp()을 사용합니다.
            xformable = UsdGeom.Xformable(origin_prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(-0.6, 0.0, 2.3))

        # 3. XRCamera 생성
        if not is_prim_path_valid(xr_camera_path):
            define_prim(xr_camera_path, "XRCamera")
            # 눈 위치는 보통 Origin(발)에서 위로 1.6m~1.7m 정도가 기본이지만,
            # 로봇 시점이라면 Origin과 거의 같은 위치(0,0,0)에 두셔도 됩니다.

        # 4. 기존 robot_camera 추가 (이미지 스트리밍/데이터 수집용)
        camera_prim_path = f"{ROBOT_PRIM_PATH}/openarm_base/robot_camera"
        camera_pos = np.array([-0.6, 0.0, 2.3]) # VR 시점과 맞추기 위해 높이 조정 추천
        camera_orientation = euler_angles_to_quats(np.array([0, 65, 0]), degrees=True)

        self.robot_camera = Camera(
            prim_path=camera_prim_path,
            position=camera_pos,
            orientation=camera_orientation,
            resolution=(720, 480) # 3 : 2 비율
        )
        self.robot_camera.initialize()
        
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
            "end_effector_frame_name": "openarm_left_link7", # [수정] left_link6 -> openarm_left_link7
            "maximum_substep_size": 0.00334
        }
        self.left_rmpflow = RmpFlow(**left_config)
        self.left_policy = ArticulationMotionPolicy(robot, self.left_rmpflow)
        
        # Right RmpFlow 설정
        right_config = {
            "robot_description_path": self.right_robot_yaml,
            "urdf_path": self.urdf_path,
            "rmpflow_config_path": self.right_rmp_config,
            "end_effector_frame_name": "openarm_right_link7", # [수정] right_hand_arm_link6 -> openarm_right_link7
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
        
        kps = np.full(robot.num_dof, 1500.0)
        kds = np.full(robot.num_dof, 180.0)
        robot.get_articulation_controller().set_gains(kps=kps, kds=kds)
        print("Robot Gains Force Set.")

    def hand_arm_Teleop(self, lvm, robot: SingleArticulation, left_joints, right_joints):
        from scipy.spatial.transform import Rotation as R
        dof_names = robot.dof_names
        l_hand_transforms, r_hand_transforms, l_wrist_mat, r_wrist_mat, head_data = lvm.get_avp_data_sync()
        
        # --- 1. Head Tracking & Front-View Mapping ---
        head_pos, head_quat = lvm.convert_head_pose(head_data)
        
        if head_pos is not None:
            # [Isaac Sim 형식 quat을 Scipy 형식 [x, y, z, w]로 변환]
            curr_head_rot = R.from_quat([head_quat[1], head_quat[2], head_quat[3], head_quat[0]])

            if not self.head_calibrated:
                print(">>> Calibrating Head View... Stand straight and look forward.")
                self.head_offset_pos = head_pos
                self.head_offset_quat = curr_head_rot # 초기 머리 회전 저장
                self.head_calibrated = True
                return None

            # A. 위치 오프셋 계산
            rel_pos = head_pos - self.head_offset_pos
            camera_base_pos = np.array([-0.6, 0.0, 2.3])
            
            # B. 상대 회전 계산: (초기 머리의 역회전 * 현재 머리 회전)
            rel_head_rot = self.head_offset_quat.inv() * curr_head_rot
            
            # C. 최종 회전 적용: (기본 50도 하방 각도 * 사용자의 상대적 머리 움직임)
            final_rot_obj = self.base_view_rot * rel_head_rot
            final_quat_scipy = final_rot_obj.as_quat()
            
            # [다시 Isaac 형식 [w, x, y, z]로 변환]
            final_quat_isaac = np.array([final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]])

            # 카메라 위치 및 회전 업데이트
            self.robot_camera.set_world_pose(
                position=camera_base_pos + rel_pos,
                orientation=final_quat_isaac 
            )

        # --- 2. Hand & Arm Pose Processing ---
        l_pos, l_rot = None, None
        r_pos, r_rot = None, None

        if l_wrist_mat is not None:
            l_pos, l_rot = lvm.convert_avp_to_isaac_pose(l_wrist_mat, 'left')
        if r_wrist_mat is not None:
            r_pos, r_rot = lvm.convert_avp_to_isaac_pose(r_wrist_mat, 'right')
            
        # 캘리브레이션 (0점 조절)
        if not self.calibrated:
            # 최소한 한 손이라도 데이터가 들어왔을 때만 캘리브레이션 진행
            if l_pos is not None or r_pos is not None:
                print(f">>> Calibrating... L: {l_pos is not None}, R: {r_pos is not None}")
                
                # 데이터가 있는 쪽만 오프셋 설정
                if l_pos is not None:
                    self.left_hand_offset_pos = l_pos
                if r_pos is not None:
                    self.right_hand_offset_pos = r_pos
                    
                self.calibrated = True
                print(">>> Calibration Complete!")
            else:
                # 데이터가 아직 없으면 계속 기다림 (True로 바꾸지 않음)
                # print("Waiting for hand data...") # 너무 많이 찍힐 수 있으니 필요시만 주석 해제
                pass
                
            return None # 캘리브레이션 전에는 아래 제어 로직을 실행하지 않음

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
        
        while simulation_app.is_running():
            # 1. 시뮬레이션이 중지되었거나 종료 중이면 루프 탈출
            if not world.is_playing():
                world.step(render=True)
                continue

            try:
                # 2. 물리 스텝 진행
                world.step(render=True)
                
                # 3. 로봇 핸들이 유효한지 최종 확인
                if not robot.handles_initialized:
                    continue

                # 4. 텔레오퍼레이션 로직 실행 (중복 호출 제거)
                action = self.hand_arm_Teleop(lvm, robot, left_joints, right_joints)
                
                if action is not None:
                    robot.apply_action(action)

                # 5. 카메라 데이터 스트리밍
                rgba_image = self.robot_camera.get_rgba()
                if rgba_image is not None and len(rgba_image.shape) == 3:
                    rgb_image = cv2.cvtColor(rgba_image[:, :, :3], cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    
                    message = json.dumps({"image": jpg_as_text})
                    if hasattr(lvm, 'send_video'):
                        lvm.send_video(message)

            except Exception as e:
                # 종료 시 발생하는 Articulation 관련 에러는 무시
                if "NoneType" in str(e) or "initialized" in str(e):
                    print("Simulation shutting down safely...")
                else:
                    print(f"Runtime Error: {e}")
                break # 에러 발생 시 루프 종료

        # 루프 종료 후 앱 완전히 닫기
        simulation_app.close()

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

    # 뷰포트 시점을 로봇 카메라로 고정
    viewport_api = viewport_utils.get_active_viewport()
    if viewport_api:
        viewport_api.camera_path = f"{primPath}/openarm_base/robot_camera"
    
    # 루프 시작 (한 번만 호출)
    loading.run_simulator(world, robot)

if __name__ == "__main__":
    main()