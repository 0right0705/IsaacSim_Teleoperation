import numpy as np
import asyncio
import websockets
import json
import ssl
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import time
from streamer import Streamer
import concurrent.futures

class TesolloVectorMapping:
    def __init__(self):
        print("[Mapping] Initializing VisionPro Streamer (WebXR)...")
        # Streamer 서버 시작
        self.streamer = Streamer(ip="192.168.10.23", port=8765)
        self.streamer.start_background_server()
        print("[Mapping] WebXR Streamer Ready. Waiting for connection...")

        # 데이터 공백 방지를 위한 마지막 데이터 저장소 (Latch)
        # 초기값은 None이 아닌 빈 리스트로 두되, 로직에서 체크함
        self.last_left_hand_json = []
        self.last_right_hand_json = []
        
        # 디버깅용 타이머
        self.last_print_time = time.time()

        # 좌표계 변환 행렬 (AVP → Isaac)
        self.R_avp_to_isaac = np.array([
            [0, 0, 1],   # Isaac X (Forward) = AVP Z  -> Roll 축 일치
            [1, 0, 0],  # Isaac Y (Left)    = -AVP X -> Pitch 축 일치
            [0, -1, 0]    # Isaac Z (Up)      = AVP Y  -> Yaw 축 일치 (이미 잘 되는 부분)
        ])  
        
        # 회전 오프셋 (기존과 동일)
        self.axis_correction = R.from_euler('z', 90, degrees=True).as_matrix()

        # 회전 오프셋 (초기 자세 정렬용)
        self.left_rotation_offset0 = R.from_euler('x', 90, degrees=True).as_matrix()
        self.left_rotation_offset1 = R.from_euler('y', -90, degrees=True).as_matrix()
        self.left_rotation_offset2 = R.from_euler('z', 90, degrees=True).as_matrix()
        self.right_rotation_offset0 = R.from_euler('x', -90, degrees=True).as_matrix()
        self.right_rotation_offset1 = R.from_euler('y', 90, degrees=True).as_matrix()
        self.right_rotation_offset2 = R.from_euler('z', 180, degrees=True).as_matrix()

    # mainHandTeleop.py 내부

    def convert_json_to_transforms(self, joints_list):
        """JSON 리스트를 (N, 4, 4) Numpy 행렬 배열로 변환합니다."""
        # [수정] Numpy array인지 먼저 체크하여 에러 방지
        if joints_list is None:
            return np.array([])
        
        if isinstance(joints_list, np.ndarray):
            if joints_list.ndim == 2: # 단일 4x4 행렬인 경우 (1, 4, 4)로 확장
                return joints_list[np.newaxis, :, :]
            return joints_list # 이미 (N, 4, 4) 형태인 경우 그대로 반환

        if len(joints_list) == 0:
            return np.array([])

        transforms = []
        for joint in joints_list:
            pos = [joint['x'], joint['y'], joint['z']]
            quat = [joint['qx'], joint['qy'], joint['qz'], joint['qw']]
            
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat(quat).as_matrix()
            mat[:3, 3] = pos
            transforms.append(mat)
        
        return np.array(transforms)

    def get_avp_data_sync(self):
        """동기식으로 데이터를 가져오며, 이미 변환된 행렬을 반환함"""
        new_data = None
        try:
            new_data = self.streamer.get_latest_data_sync(timeout=0.001)
        except Exception:
            new_data = None

        if new_data and isinstance(new_data, list):
            for hand in new_data:
                if hand['hand'] == 'Left':
                    self.last_left_hand_json = hand['joints']
                elif hand['hand'] == 'Right':
                    self.last_right_hand_json = hand['joints']

        # 여기서 이미 convert_json_to_transforms를 호출함
        left_hand_transforms = self.convert_json_to_transforms(self.last_left_hand_json)
        right_hand_transforms = self.convert_json_to_transforms(self.last_right_hand_json)
        
        # Wrist 행렬 추출 (이미 4x4 행렬임)
        left_wrist_mat = left_hand_transforms[0] if len(left_hand_transforms) > 0 else None
        right_wrist_mat = right_hand_transforms[0] if len(right_hand_transforms) > 0 else None

        return left_hand_transforms, right_hand_transforms, left_wrist_mat, right_wrist_mat


    def convert_avp_to_isaac_pose(self, avp_matrix, side="left"):
        if avp_matrix is None or (isinstance(avp_matrix, np.ndarray) and avp_matrix.size == 0):
            return None, None
        
        avp_matrix = np.array(avp_matrix)
        if avp_matrix.ndim == 3: avp_matrix = avp_matrix[0]
        
        # 1. 원본 데이터 추출
        rot_avp = avp_matrix[:3, :3]
        pos_avp = avp_matrix[:3, 3]

        # 2. [기저 변환] AVP의 회전 축 정의를 Isaac 축 정의로 변환
        # 이 연산을 통해 "X축 회전"이 Isaac에서도 올바른 방향의 회전이 됩니다.
        # rot_mapped = self.R_avp_to_isaac @ rot_avp @ self.R_avp_to_isaac.T
        
        pos_isaac = self.R_avp_to_isaac @ pos_avp
        M = self.R_avp_to_isaac
        rot_mapped = M @ rot_avp @ M.T
        
        if side == "left":
            # 맵핑된 회전 뒤에 축 교정(axis_correction)을 먼저 곱해 축 정의를 바꾼 후 offset 적용
            rot_final = rot_mapped @ self.left_rotation_offset1 
        else:
            rot_final = rot_mapped  @ self.right_rotation_offset1 @ self.right_rotation_offset2
        # 4. 위치(Position) 변환
        # pos_isaac = self.R_avp_to_isaac @ pos_avp
        # 5. 쿼터니언 변환 (Isaac: [w, x, y, z])
        quat_scipy = R.from_matrix(rot_final).as_quat()
        quat_isaac = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        return pos_isaac, quat_isaac

    # def convert_avp_to_isaac_pose(self, avp_matrix, side="left"):
    #     # 입력이 None이거나 빈 배열이면 패스
    #     if avp_matrix is None or (isinstance(avp_matrix, np.ndarray) and avp_matrix.size == 0):
    #         return None, None
            
    #     avp_matrix = np.array(avp_matrix)

    #     # 가끔 (1, 4, 4) 형태로 들어올 수 있음
    #     if avp_matrix.ndim == 3 and avp_matrix.shape[0] == 1:
    #         avp_matrix = avp_matrix[0]
        
    #     # 4x4 행렬인지 확인
    #     if avp_matrix.shape != (4, 4):
    #         return None, None

    #     # [Rotation 처리]
    #     rot_avp = avp_matrix[:3, :3]

        
    #     # 1단계: 월드 좌표축 변환
    #     # rot_isaac_base = self.R_avp_to_isaac @ rot_avp 
    #     rot_isaac_base = rot_avp
        
    #     if side == "left":
    #         rot_final = rot_avp @ self.left_rotation_offset @ self.axis_swap
    #         # rot_final = rot_isaac_base
    #     else: # right
    #         rot_final = rot_avp @ self.right_rotation_offset @ self.axis_swap
    #         # rot_final = rot_isaac_base

    #     # [Position 처리] 
    #     pos_avp = avp_matrix[:3, 3]
    #     # pos_isaac = self.R_avp_to_isaac @ pos_avp 
    #     pos_isaac = pos_avp
        
    #     # 최종 회전(rot_final)을 쿼터니언으로 변환
    #     quat_scipy = R.from_matrix(rot_final).as_quat() # x, y, z, w
    #     quat_isaac = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]) # w, x, y, z
        
    #     return pos_isaac, quat_isaac
    
    def calculate_hand_angles(self, transforms):
        WRIST = 0 
        chains = {
            "Thumb": [WRIST, 1, 2, 3, 4],       
            "Index": [WRIST, 5, 6, 7, 8],    
            "Middle": [WRIST, 10, 11, 12, 13], 
            "Ring":   [WRIST, 15, 16, 17, 18],
            "Little" : [WRIST, 20, 21, 22, 23]  
        }
        
        results = {}
        # transforms가 빈 배열이면 빈 결과 반환
        if len(transforms) == 0:
            return results

        for name, indices in chains.items():
            angles = []
            for i in range(len(indices) - 1):
                p_idx = indices[i] 
                c_idx = indices[i+1] 
                
                # 인덱스 초과 체크
                if c_idx >= len(transforms): continue
                
                # [중요] transforms가 이제 4x4 행렬 리스트이므로 인덱싱 가능
                R_parent = transforms[p_idx][:3, :3] 
                R_child = transforms[c_idx][:3, :3] 
                
                R_rel = np.dot(R_parent.T, R_child)
                euler = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
                
                joint_name = f"{p_idx}->{c_idx}"
                if p_idx == WRIST:
                    joint_name += " (Base/Meta)"
                    
                angles.append({
                    "joint": joint_name,
                    "xyz": euler
                })
            results[name] = angles
        return results
    
    def get_joint_index(self, dof_names):
        """
        제공된 dof_names 리스트에서 로봇 손가락 관절의 정확한 인덱스를 추출합니다.
        """
        left_joint_idx = []
        right_joint_idx = []
        
        # 0번부터 15번까지의 관절을 정확한 이름으로 찾아서 리스트에 담습니다.
        for i in range(1, 6):
            for j in range(1, 5):
                l_name = f'lj_dg_{i}_{j}'
                r_name = f'rj_dg_{i}_{j}'
                
                if l_name in dof_names:
                    left_joint_idx.append(dof_names.index(l_name))
                else:
                    print(f"[Warning] Joint {l_name} not found in robot.dof_names")
                    left_joint_idx.append(None)
                    
                if r_name in dof_names:
                    right_joint_idx.append(dof_names.index(r_name))
                else:
                    print(f"[Warning] Joint {r_name} not found in robot.dof_names")
                    right_joint_idx.append(None)
                
        return left_joint_idx, right_joint_idx

    def map_to_left_robot_action(self, angles_dict, dof_names, left_joints): 
        actions = np.zeros(len(dof_names))
        if not angles_dict: return actions

        # [수정] 부호 반전 및 인덱스 고정
        def val(deg_array, scale=1.0):
            # 데이터가 [-23, 4, 1] 식으로 들어오므로 0번 인덱스가 메인 관절각입니다.
            # 음수 값이 굽힘을 나타내므로 앞에 -를 붙여 양수로 변환합니다.
            return np.deg2rad(-deg_array[0]) * scale 

        FLEXION_SCALE = 1.4
        
        def safe_set(idx_in_list, value):
            if idx_in_list is not None:
                actions[idx_in_list] = value


        # 1. Thumb (엄지) - Tesollo lj_dg_1_1 ~ 1_4 
        if "Thumb" in angles_dict:
            avp_thb = angles_dict["Thumb"]
            # 엄지는 관절 구조상 데이터 매핑 확인이 가장 중요함
            safe_set(left_joints[0], val(avp_thb[0]['xyz'], 1.5)) # lj_dg_1_1
            safe_set(left_joints[1], val(avp_thb[1]['xyz'], 1.5)) # lj_dg_1_2
            safe_set(left_joints[2], val(avp_thb[1]['xyz'], -FLEXION_SCALE)) # lj_dg_1_3
            safe_set(left_joints[3], val(avp_thb[2]['xyz'], -FLEXION_SCALE)) # lj_dg_1_4
            
        # 2. Index (검지) - Tesollo lj_dg_2_1 ~ 2_4 
        if "Index" in angles_dict:
            avp_idx = angles_dict["Index"]
            if len(avp_idx) >= 3:
                safe_set(left_joints[4], val(avp_idx[0]['xyz'], 0.5))           # Spread (lj_dg_2_1)
                safe_set(left_joints[5], val(avp_idx[1]['xyz'], FLEXION_SCALE)) # Flexion 1 (lj_dg_2_2)
                safe_set(left_joints[6], val(avp_idx[2]['xyz'], FLEXION_SCALE)) # Flexion 2 (lj_dg_2_3)
                safe_set(left_joints[7], val(avp_idx[3]['xyz'], FLEXION_SCALE)) # Flexion 3 (lj_dg_2_4)

        # 3. Middle (중지) - Tesollo lj_dg_3_1 ~ 3_4 
        if "Middle" in angles_dict:
            avp_mid = angles_dict["Middle"]
            if len(avp_mid) >= 3:
                safe_set(left_joints[8], val(avp_mid[0]['xyz'], 0.5))
                safe_set(left_joints[9], val(avp_mid[1]['xyz'], FLEXION_SCALE))
                safe_set(left_joints[10], val(avp_mid[2]['xyz'], FLEXION_SCALE))
                safe_set(left_joints[11], val(avp_mid[3]['xyz'], FLEXION_SCALE))

        # 4. Ring (약지) - Tesollo lj_dg_4_1 ~ 4_4 
        if "Ring" in angles_dict:
            avp_ring = angles_dict["Ring"]
            if len(avp_ring) >= 3:
                safe_set(left_joints[12], val(avp_ring[0]['xyz'], 0.5))
                safe_set(left_joints[13], val(avp_ring[1]['xyz'], FLEXION_SCALE))
                safe_set(left_joints[14], val(avp_ring[2]['xyz'], FLEXION_SCALE))
                safe_set(left_joints[15], val(avp_ring[3]['xyz'], FLEXION_SCALE))

        # 5. Little (소지/새끼손가락) - Tesollo lj_dg_5_1 ~ 5_4 
        if "Little" in angles_dict:
            avp_lit = angles_dict["Little"]
            if len(avp_lit) >= 3:
                safe_set(left_joints[16], val(avp_lit[0]['xyz'], 0.5))
                safe_set(left_joints[17], val(avp_lit[1]['xyz'], FLEXION_SCALE))
                safe_set(left_joints[18], val(avp_lit[2]['xyz'], FLEXION_SCALE))
                safe_set(left_joints[19], val(avp_lit[3]['xyz'], FLEXION_SCALE))

        return actions

    def map_to_right_robot_action(self, angles_dict, dof_names, right_joints): 
        actions = np.zeros(len(dof_names))
        if not angles_dict: return actions

        def val(deg_array, scale=1.0):
            return np.deg2rad(-deg_array[0]) * scale 

        FLEXION_SCALE = 1.4 
        def safe_set(idx_in_list, value):
            if idx_in_list is not None:
                actions[idx_in_list] = value

        # 1. Thumb (엄지) - right_joints[0~3]
        if "Thumb" in angles_dict:
            avp_thb = angles_dict["Thumb"]
            safe_set(right_joints[0], val(avp_thb[0]['xyz'], -1.5))
            safe_set(right_joints[1], val(avp_thb[1]['xyz'], -1.5))
            safe_set(right_joints[2], val(avp_thb[1]['xyz'], FLEXION_SCALE))
            safe_set(right_joints[3], val(avp_thb[2]['xyz'], FLEXION_SCALE))
            
        # 2. Index (검지) - right_joints[4~7]
        if "Index" in angles_dict:
            avp_idx = angles_dict["Index"]
            if len(avp_idx) >= 3:
                safe_set(right_joints[4], val(avp_idx[0]['xyz'], 0.5))
                safe_set(right_joints[5], val(avp_idx[1]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[6], val(avp_idx[2]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[7], val(avp_idx[3]['xyz'], FLEXION_SCALE))

        # 3. Middle (중지) - right_joints[8~11]
        if "Middle" in angles_dict:
            avp_mid = angles_dict["Middle"]
            if len(avp_mid) >= 3:
                safe_set(right_joints[8], val(avp_mid[0]['xyz'], 0.5))
                safe_set(right_joints[9], val(avp_mid[1]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[10], val(avp_mid[2]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[11], val(avp_mid[3]['xyz'], FLEXION_SCALE))

        # 4. Ring (약지) - right_joints[12~15]
        if "Ring" in angles_dict:
            avp_ring = angles_dict["Ring"]
            if len(avp_ring) >= 3:
                safe_set(right_joints[12], val(avp_ring[0]['xyz'], 0.5))
                safe_set(right_joints[13], val(avp_ring[1]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[14], val(avp_ring[2]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[15], val(avp_ring[3]['xyz'], FLEXION_SCALE))

        # 5. Little (소지) - right_joints[16~19]
        if "Little" in angles_dict:
            avp_lit = angles_dict["Little"]
            if len(avp_lit) >= 3:
                safe_set(right_joints[16], val(avp_lit[0]['xyz'], 0.5))
                safe_set(right_joints[17], val(avp_lit[1]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[18], val(avp_lit[2]['xyz'], FLEXION_SCALE))
                safe_set(right_joints[19], val(avp_lit[3]['xyz'], FLEXION_SCALE))

        return actions