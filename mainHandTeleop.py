import numpy as np
from avp_stream import VisionProStreamer
from scipy.spatial.transform import Rotation as R
import time

AVP_IP = "192.168.10.224"


class LeapVectorMapping:
    def __init__(self):
        print("[Mapping] Initializing VisionPro Streamer...")
        self.vps = VisionProStreamer(ip=AVP_IP, record=True)
        print("[Mapping] Vector-based Mapper Ready.")

        # 좌표계 변환 행렬 (AVP → Isaac)
        self.R_avp_to_isaac = np.array([
            [1, 0, 0],  # Isaac X(앞) = AVP Z(뒤)의 반대 (-Z)
            [0, 1, 0],  # Isaac Y(좌) = AVP X(우)의 반대 (-X)
            [0, 0, 1]    # Isaac Z(위) = AVP Y(위) 그대로 (+Y)
        ])  # right-handed → Z-forward system
        # 왼손 손목 꺾여있는거 rotation
        self.left_rotation_offset = R.from_euler('y', 90, degrees=True).as_matrix()
        self.left_rotation_offset2 = R.from_euler('x', 180, degrees=True).as_matrix()
        # 오른손 손목 꺾여있는거 rotation
        self.right_rotation_offset = R.from_euler('y', -90, degrees=True).as_matrix()
        # self.hand_rotation_offset = np.eye(3)
        # neutral offset (캘리브레이션 필요)
        self.left_offset = np.zeros(16)
        self.right_offset = np.zeros(16)
        self.offset_applied = False

    def transform(self, p):
        # AVP 좌표 → Isaac 좌표 변환
        return (self.R_avp_to_isaac @ p.reshape(3, )).reshape(3, )

    def get_avp_data(self):
        r = self.vps.latest
        left_hand = np.array(r["left_fingers"])
        right_hand = np.array(r["right_fingers"])
        left_wrist = np.array(r["left_wrist"])
        right_wrist = np.array(r["right_wrist"])
        return left_hand, right_hand, left_wrist, right_wrist

    def convert_avp_to_isaac_pose(self, avp_matrix, side="left"):
        # 1. 입력 확인
        if avp_matrix is None:
            return None, None
            
        avp_matrix = np.array(avp_matrix)

        if avp_matrix.ndim == 3 and avp_matrix.shape[0] == 1:
            avp_matrix = avp_matrix[0]

        # [Rotation 처리]
        rot_avp = avp_matrix[:3, :3]
        
        # 1단계: 월드 좌표축 변환
        rot_isaac_base = self.R_avp_to_isaac @ rot_avp 
        
        if side == "left":
            rot_final = rot_isaac_base @ self.left_rotation_offset2 @ self.left_rotation_offset 
        else: # right
            rot_final = rot_isaac_base @ self.right_rotation_offset

        # [Position 처리] 
        pos_avp = avp_matrix[:3, 3]
        pos_isaac = self.R_avp_to_isaac @ pos_avp 
        
        # 최종 회전(rot_final)을 쿼터니언으로 변환
        quat_scipy = R.from_matrix(rot_final).as_quat() # x, y, z, w
        quat_isaac = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]) # w, x, y, z
        
        return pos_isaac, quat_isaac
    
    def calculate_hand_angles(self, transforms):
        WRIST = 0 
        chains = {
            # 엄지: Wrist -> Knuckle(1) -> InterBase(2) -> InterTip(3)
            "Thumb": [WRIST, 1, 2, 3],       
            # 검지: Wrist -> Metacarpal(5) -> Knuckle(6) -> InterBase(7) -> InterTip(8)
            "Index": [WRIST, 5, 6, 7, 8],    
            # 중지: Wrist -> Metacarpal(10) -> Knuckle(11) -> InterBase(12) -> InterTip(13)
            "Middle": [WRIST, 10, 11, 12, 13], 
            # 약지: Wrist -> Metacarpal(15) -> Knuckle(16) -> InterBase(17) -> InterTip(18)
            "Ring":   [WRIST, 15, 16, 17, 18]  
        }
        
        results = {}
        
        for name, indices in chains.items():
            angles = []
            for i in range(len(indices) - 1):
                p_idx = indices[i] # parent index
                c_idx = indices[i+1] # child index
                
                if c_idx >= len(transforms): continue
                
                # 회전 행렬 추출
                R_parent = transforms[p_idx][:3, :3] # parent rotation
                R_child = transforms[c_idx][:3, :3] # child rotation
                
                # 상대 회전: Parent_Inv * Child
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
        def get_idx(name): # dof_names에서 이름 찾아 그 이름의 인덱스 번호 리턴. 조인트 이름에 맞춰서 바꾸면 됨
            try: return dof_names.index(name)
            except ValueError: return None
        left_joint_idx = []
        right_joint_idx = []
        left_prefix = 'left_' # 용현님거 기준
        right_prefix = 'right_' # 용현님거 기준
        idxNum = get_idx(f'{left_prefix}0')
        if idxNum is None:
            left_prefix = 'j' # 내거 기준
            right_prefix = 'rj' # 내거 기준
        for i in range(0, 16):
            l_name = f'{left_prefix}{i}'
            r_name = f'{right_prefix}{i}'
            
            # 이름을 인덱스 숫자로 변환
            l_idx = get_idx(l_name)
            r_idx = get_idx(r_name)
            
            # 인덱스가 None이면(이름을 못 찾으면) 에러 방지를 위해 0이나 -1 처리, 혹은 에러 출력
            if l_idx is None: print(f"[Warning] Joint {l_name} not found!")
            if r_idx is None: print(f"[Warning] Joint {r_name} not found!")

            left_joint_idx.append(l_idx)
            right_joint_idx.append(r_idx)
            
        return left_joint_idx, right_joint_idx


    
    def map_to_left_robot_action(self, angles_dict, dof_names, left_joints): #왼손용
        actions = np.zeros(len(dof_names))
        main_start_time = time.perf_counter()

        def val(deg, scale=1.0):
            return np.deg2rad(deg) * scale
        
        FLEXION_SCALE = 1.4 # 손등을 보고 있을 때 손가락 굽힘이 잘 안되는 것 방지용

        # 검지 (Index)
        index_start_time = time.perf_counter()
        avp_idx = angles_dict["Index"]
        
        idx_j0 = left_joints[0]
        actions[idx_j0] = val(avp_idx[1]['xyz'][1], scale=0.5)

        idx_j1 = left_joints[1]
        actions[idx_j1] = val(avp_idx[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j2 = left_joints[2]
        actions[idx_j2] = val(avp_idx[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j3 = left_joints[3]
        actions[idx_j3] = val(avp_idx[3]['xyz'][2], scale=FLEXION_SCALE)
        index_end_time = time.perf_counter()
        index_elapsed_time = index_end_time - index_start_time
        # print(f'index searching time : {index_elapsed_time:.7f} 초\n')

        # 중지 (Middle)
        avp_mid = angles_dict["Middle"]
        
        idx_j4 = left_joints[4]
        actions[idx_j4] = val(avp_mid[1]['xyz'][1], 0.5)

        idx_j5 = left_joints[5]
        actions[idx_j5] = val(avp_mid[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j6 = left_joints[6]
        actions[idx_j6] = val(avp_mid[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j7 = left_joints[7]
        actions[idx_j7] = val(avp_mid[3]['xyz'][2], scale=FLEXION_SCALE)

        # 약지 (Ring)
        avp_ring = angles_dict["Ring"]
        
        idx_j8 = left_joints[8]
        actions[idx_j8] = val(avp_ring[1]['xyz'][1], 0.5)

        idx_j9 = left_joints[9]
        actions[idx_j9] = val(avp_ring[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j10 = left_joints[10]
        actions[idx_j10] = val(avp_ring[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j11 = left_joints[11]
        actions[idx_j11] = val(avp_ring[3]['xyz'][2], scale=FLEXION_SCALE)

        # 엄지 (Thumb)
        # j12는 엄지 맨 밑 로테이션 (이거 없으면 핀치시 엄지가 안쪽으로 로테이션 안됨)
        # j14는 엄지가 일자로 안 펴지는거 피는용도

        avp_thb = angles_dict["Thumb"]

        THUMB_OFFSET = 20.0
        idx_j12 = left_joints[12]
        actions[idx_j12] = val(avp_thb[0]['xyz'][1] - THUMB_OFFSET, scale=3.0)

        idx_j13 = left_joints[13]
        actions[idx_j13] =  val(avp_thb[1]['xyz'][1])

        idx_j14 = left_joints[14]
        actions[idx_j14] = val(avp_thb[1]['xyz'][2]+ THUMB_OFFSET, scale=3.0) # 엄지 굽힘은 이 값으로 해결

        idx_j15 = left_joints[15]
        actions[idx_j15] = val(avp_thb[2]['xyz'][2])
        main_end_time = time.perf_counter()
        main_elapsed_time = main_end_time - main_start_time
        # print(f'main 코드 실행 시간 : {main_elapsed_time:.5f} 초\n')
        # print('\n')
        return actions
    
    def map_to_right_robot_action(self, angles_dict, dof_names, right_joint): #오른손용
        actions = np.zeros(len(dof_names))

        def get_idx(name): # dof_names에서 이름 찾아 그 이름의 인덱스 번호 리턴. 조인트 이름에 맞춰서 바꾸면 됨
            try: return dof_names.index(name)
            except ValueError: return None

        def val(deg, scale=1.0):
            return np.deg2rad(deg) * scale
        
        FLEXION_SCALE = 1.4 # 손등을 보고 있을 때 손가락 굽힘이 잘 안되는 것 방지용

        # 검지 (Index)
        avp_idx = angles_dict["Index"]
        
        idx_j0 = right_joint[0]
        actions[idx_j0] = val(avp_idx[1]['xyz'][1], scale=0.5)

        idx_j1 = right_joint[1]
        actions[idx_j1] = val(avp_idx[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j2 = right_joint[2]
        actions[idx_j2] = val(avp_idx[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j3 = right_joint[3]
        actions[idx_j3] = val(avp_idx[3]['xyz'][2], scale=FLEXION_SCALE)

        # 중지 (Middle)
        avp_mid = angles_dict["Middle"]
        
        idx_j4 = right_joint[4]
        actions[idx_j4] = val(avp_mid[1]['xyz'][1], 0.5)

        idx_j5 = right_joint[5]
        actions[idx_j5] = val(avp_mid[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j6 = right_joint[6]
        actions[idx_j6] = val(avp_mid[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j7 = right_joint[7]
        actions[idx_j7] = val(avp_mid[3]['xyz'][2], scale=FLEXION_SCALE)

        # 약지 (Ring)
        avp_ring = angles_dict["Ring"]
        
        idx_j8 = right_joint[8]
        actions[idx_j8] = val(avp_ring[1]['xyz'][1], 0.5)

        idx_j9 = right_joint[9]
        actions[idx_j9] = val(avp_ring[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j10 = right_joint[10]
        actions[idx_j10] = val(avp_ring[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j11 = right_joint[11]
        actions[idx_j11] = val(avp_ring[3]['xyz'][2], scale=FLEXION_SCALE)

        # 엄지 (Thumb)
        # rj12는 엄지 맨 밑 로테이션 (이거 없으면 핀치시 엄지가 안쪽으로 로테이션 안됨)
        # rj14는 엄지가 일자로 안 펴지는거 피는용도
        avp_thb = angles_dict["Thumb"]
        THUMB_OFFSET = 20.0
        idx_j12 = right_joint[12]
        actions[idx_j12] = val(avp_thb[0]['xyz'][1] + THUMB_OFFSET, scale=-3.0)

        idx_j13 = right_joint[13]
        actions[idx_j13] =  val(avp_thb[1]['xyz'][1])

        idx_j14 = right_joint[14]
        actions[idx_j14] = val(avp_thb[1]['xyz'][2], scale=3.0) # 엄지 굽힘은 이 값으로 해결

        idx_j15 = right_joint[15]
        actions[idx_j15] = val(avp_thb[2]['xyz'][2])

        return actions