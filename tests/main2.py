import argparse
import numpy as np
import os

# [중요] SimulationApp 실행
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim & Core 라이브러리
from isaacsim.core.api import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim, is_prim_path_valid
from omni.isaac.core.utils.viewports import set_camera_view

# [NEW] Lula RMPflow 관련 모듈
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

# 사용자 모듈 (기존 avp_stream 관련 등)
from avp_stream import VisionProStreamer
from scipy.spatial.transform import Rotation as R

AVP_IP = "192.168.10.224"

class LeapVectorMapping:
    def __init__(self):
        print("[Mapping] Initializing VisionPro Streamer...")
        self.vps = VisionProStreamer(ip=AVP_IP, record=True)
        print("[Mapping] Vector-based Mapper Ready.")

        # 좌표계 변환 행렬 (AVP → Isaac)
        # Vision Pro: Y-up (혹은 설정에 따라 다름), Isaac: Z-up
        # 사용자의 기존 변환 행렬 유지
        self.R_avp_to_isaac = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

    def transform_pos(self, p):
        """위치 벡터 변환 (3,)"""
        return (self.R_avp_to_isaac @ p.reshape(3, )).reshape(3, )

    def transform_matrix(self, matrix_4x4):
        """
        4x4 변환 행렬 전체를 Isaac 좌표계로 변환
        Vision Pro의 Pose 행렬을 로봇 베이스 기준의 Target Pose로 변환합니다.
        """
        # 1. 회전 변환
        rot_avp = matrix_4x4[:3, :3]
        pos_avp = matrix_4x4[:3, 3]

        # R_isaac = R_transform * R_avp * R_transform.T (기저 변환)
        # 혹은 단순히 방향만 맞춘다면: R_isaac = R_transform @ R_avp
        # 여기서는 사용자가 정의한 좌표축 변경을 적용
        rot_isaac = self.R_avp_to_isaac @ rot_avp @ self.R_avp_to_isaac.T
        pos_isaac = self.R_avp_to_isaac @ pos_avp
        
        # 스케일링 (Vision Pro 미터 단위 -> Isaac 미터 단위, 보통 1:1이지만 필요시 조정)
        SCALE_FACTOR = 1.0 
        pos_isaac *= SCALE_FACTOR

        # 오프셋 (로봇 베이스 위치에 따라 조정 필요, 여기서는 로봇 앞쪽으로 이동)
        # 예: 로봇 앞으로 0.5m, 위로 0.5m 지점을 원점으로 사용
        pos_offset = np.array([0.5, 0.0, 0.3]) 
        
        target_matrix = np.eye(4)
        target_matrix[:3, :3] = rot_isaac
        target_matrix[:3, 3] = pos_isaac + pos_offset
        
        return target_matrix

    def get_avp_data(self):
        return self.vps.latest

    def calculate_hand_angles(self, transforms):
        # ... (기존 손가락 각도 계산 코드 그대로 유지) ...
        WRIST = 0 
        chains = {
            "Thumb (엄지)": [WRIST, 1, 2, 3],       
            "Index (검지)": [WRIST, 5, 6, 7, 8],    
            "Middle (중지)": [WRIST, 10, 11, 12, 13], 
            "Ring (약지)":   [WRIST, 15, 16, 17, 18]  
        }
        results = {}
        for name, indices in chains.items():
            angles = []
            for i in range(len(indices) - 1):
                p_idx = indices[i]
                c_idx = indices[i+1]
                if c_idx >= len(transforms): continue
                R_parent = transforms[p_idx][:3, :3]
                R_child = transforms[c_idx][:3, :3]
                R_rel = np.dot(R_parent.T, R_child)
                euler = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
                joint_name = f"{p_idx}->{c_idx}"
                angles.append({"joint": joint_name, "xyz": euler})
            results[name] = angles
        return results
    
    def map_to_hand_action(self, angles_dict, dof_names, side="left"):
        # ... (기존 map_to_robot_action 함수 내용을 이름만 변경) ...
        # 기존 코드의 내용을 여기에 그대로 사용합니다.
        # 편의상 생략된 부분은 위에서 제공해주신 코드와 동일하다고 가정합니다.
        actions = np.zeros(len(dof_names))
        # (기존 로직 복사 붙여넣기 필요)
        # 아래는 예시로 0으로 채움, 실제 사용시 위 코드 내용을 복사하세요.
        return actions 

class RobotController:
    def __init__(self, world: World, robot: Articulation, urdf_path, rmp_config_path, ee_frame_name):
        self.world = world
        self.robot = robot
        self.mapper = LeapVectorMapping()
        
        # [NEW] RMPflow 설정
        self.rmp_flow = RmpFlow(
            robot_description_path=urdf_path,
            urdf_path=urdf_path,
            rmpflow_config_path=rmp_config_path,
            end_effector_frame_name=ee_frame_name,
            maximum_substep_size=0.00334
        )
        
        # RMPflow 디버그 시각화 활성화
        self.rmp_flow.set_ignore_state_updates(True)
        self.rmp_flow.visualize_collision_spheres()
        self.rmp_flow.visualize_end_effector_position()
        
        self.physics_dt = 1.0 / 60.0
        self.articulation_rmp = ArticulationMotionPolicy(robot, self.rmp_flow, self.physics_dt)

    def run(self):
        dof_names = self.robot.dof_names
        print(f"Robot DoF Names: {dof_names}")
        
        # RMPFlow가 제어하는 관절의 인덱스를 찾습니다.
        # (설정 파일에 정의된 active joint만 제어됨)
        
        while simulation_app.is_running():
            # 1. Vision Pro 데이터 수신
            avp_data = self.mapper.get_avp_data()
            if not avp_data or "left_wrist" not in avp_data:
                self.world.step(render=True)
                continue

            # -------------------------------------------------
            # [Part A] Arm Control (RMPflow)
            # -------------------------------------------------
            # Vision Pro의 Wrist Matrix 가져오기 (4x4 가정)
            # 만약 stream이 position(3,)만 준다면 회전은 고정해야 함
            raw_wrist_matrix = np.array(avp_data["left_wrist"]) 
            
            # Isaac 좌표계로 타겟 변환
            target_pose = self.mapper.transform_matrix(raw_wrist_matrix)
            target_pos = target_pose[:3, 3]
            target_rot = target_pose[:3, :3] # Matrix
            
            # Matrix -> Quaternion (w, x, y, z) 변환
            # scipy 사용:
            quat_scipy = R.from_matrix(target_rot).as_quat() 
            # Isaac Sim RmpFlow는 (w, x, y, z) 순서를 사용함. 
            # scipy는 (x, y, z, w) 이므로 순서 변경 필요!
            target_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

            # RMPFlow 타겟 설정
            self.rmp_flow.set_end_effector_motion_policy_prior_state(
                position=target_pos,
                orientation=target_quat,
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3)
            )

            # RMP Step 계산 (현재 로봇 상태 기반)
            # action은 ArticulationAction 객체임
            rmp_action = self.articulation_rmp.get_next_articulation_action()
            
            # -------------------------------------------------
            # [Part B] Hand Control (Finger Mapping)
            # -------------------------------------------------
            left_fingers = np.array(avp_data["left_fingers"])
            finger_angles = self.mapper.calculate_hand_angles(left_fingers)
            
            # 매핑 함수 이름이 map_to_robot_action이라고 가정 (사용자 코드 참조)
            # 주의: 여기서 dof_names 전체 길이만큼의 0 배열에 손가락 값만 채워져서 나옴
            hand_action_values = self.mapper.map_to_robot_action(finger_angles, dof_names, side="left")
            
            # -------------------------------------------------
            # [Part C] Action Merging (Arm + Hand)
            # -------------------------------------------------
            # RMP가 계산한 팔 관절 값 + 매핑된 손가락 관절 값 합치기
            
            final_joint_positions = np.zeros(self.robot.num_dof)
            
            # 1. RMP 결과 적용 (RMP는 전체 관절 중 설정된 관절값만 반환하거나 전체를 반환함)
            # rmp_action.joint_positions가 None이 아닐 때
            if rmp_action.joint_positions is not None:
                # 인덱스 매칭이 중요함. ArticulationMotionPolicy가 자동으로 매칭해줌.
                # 하지만 수동 합성을 위해 joint_indices를 확인해야 할 수 있음.
                # 여기선 간단히 RMP가 전체 DoF에 대한 값을 관리한다고 가정하거나,
                # rmp_action.joint_indices를 이용해 마스킹합니다.
                
                if rmp_action.joint_indices is not None:
                    final_joint_positions[rmp_action.joint_indices] = rmp_action.joint_positions
                else:
                    # 인덱스가 없으면 전체 길이라고 가정 (위험할 수 있음)
                    final_joint_positions = rmp_action.joint_positions

            # 2. Hand 결과 덮어쓰기
            # 손가락 관절 인덱스만 찾아서 hand_action_values의 값을 적용
            # map_to_robot_action이 0이 아닌 값을 가진 곳(유효한 손가락 관절)만 업데이트
            
            for i, val in enumerate(hand_action_values):
                # 값이 0이 아니거나, 해당 인덱스가 손가락 관절 이름에 해당하면 덮어쓰기
                # 여기서는 간단히 0이 아닌 경우 덮어쓰는 방식 사용 (Calibration 주의)
                # 더 정확히는 손가락 관절 인덱스 목록을 미리 정의하는 것이 좋음
                if abs(val) > 1e-4: 
                    final_joint_positions[i] = val

            # 로봇에 최종 명령 전달
            self.robot.apply_action(ArticulationAction(joint_positions=final_joint_positions))
            
            self.world.step(render=True)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    world = World(stage_units_in_meters=1.0)
    
    # ---------------------------------------------------------
    # [설정] 파일 경로 (사용자 환경에 맞게 수정 필수)
    # ---------------------------------------------------------
    # USD 파일 (시각적/물리적 모델)
    usd_path = "/home/youngwoo/newTeleop/assets/full_robot/full_robot.usd"
    
    # URDF 파일 (RMPFlow가 키네마틱스 계산용으로 사용)
    urdf_path = "/home/youngwoo/newTeleop/assets/full_robot/full_robot.urdf"
    
    # RMPFlow 설정 파일 (충돌 구체, 관절 한계 등 정의)
    rmp_config_path = "/home/youngwoo/newTeleop/assets/full_robot/left_hand.yaml"
    
    # 로봇의 엔드 이펙터(손목) 링크 이름 (URDF 상의 이름)
    # 예: "wrist_roll_link", "link_6", "ee_link" 등 확인 필요
    ee_frame_name = "wrist_roll_link" 
    
    # 로봇 Prim Path
    robot_prim_path = "/World/full_robot"

    # Scene Setup
    add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim_path)
    robot = Articulation(prim_path=robot_prim_path, name="Robot")
    world.scene.add(robot)
    
    # 빛 추가
    if not is_prim_path_valid("/World/DomeLight"):
        define_prim("/World/DomeLight", "DomeLight")

    world.reset()
    world.render()
    set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.5])
    
    # 컨트롤러 실행
    controller = RobotController(world, robot, urdf_path, rmp_config_path, ee_frame_name)
    controller.run()

if __name__ == "__main__":
    main()