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
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])  # right-handed → Z-forward system

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
        left_wrist_roll = np.array(r["left_wrist_roll"])
        right_wrist_roll = np.array(r["right_wrist_roll"])
        return left_hand, right_hand

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
    
    def map_to_left_robot_action(self, angles_dict, dof_names): #왼손용
        actions = np.zeros(len(dof_names))
        main_start_time = time.perf_counter()
        def get_idx(name): # dof_names에서 이름 찾아 그 이름의 인덱스 번호 리턴. 조인트 이름에 맞춰서 바꾸면 됨
            try: return dof_names.index(name)
            except ValueError: return None

        def val(deg, scale=1.0):
            return np.deg2rad(deg) * scale
        
        FLEXION_SCALE = 1.4 # 손등을 보고 있을 때 손가락 굽힘이 잘 안되는 것 방지용

        # 검지 (Index)
        index_start_time = time.perf_counter()
        avp_idx = angles_dict["Index"]
        
        idx_j0 = get_idx("j0")
        actions[idx_j0] = val(avp_idx[1]['xyz'][1], scale=0.5)

        idx_j1 = get_idx("j1")
        actions[idx_j1] = val(avp_idx[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j2 = get_idx("j2")
        actions[idx_j2] = val(avp_idx[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j3 = get_idx("j3")
        actions[idx_j3] = val(avp_idx[3]['xyz'][2], scale=FLEXION_SCALE)
        index_end_time = time.perf_counter()
        index_elapsed_time = index_end_time - index_start_time
        # print(f'index searching time : {index_elapsed_time:.7f} 초\n')

        # 중지 (Middle)
        avp_mid = angles_dict["Middle"]
        
        idx_j4 = get_idx("j4")
        actions[idx_j4] = val(avp_mid[1]['xyz'][1], 0.5)

        idx_j5 = get_idx("j5")
        actions[idx_j5] = val(avp_mid[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j6 = get_idx("j6")
        actions[idx_j6] = val(avp_mid[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j7 = get_idx("j7")
        actions[idx_j7] = val(avp_mid[3]['xyz'][2], scale=FLEXION_SCALE)

        # 약지 (Ring)
        avp_ring = angles_dict["Ring"]
        
        idx_j8 = get_idx("j8")
        actions[idx_j8] = val(avp_ring[1]['xyz'][1], 0.5)

        idx_j9 = get_idx("j9")
        actions[idx_j9] = val(avp_ring[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j10 = get_idx("j10")
        actions[idx_j10] = val(avp_ring[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j11 = get_idx("j11")
        actions[idx_j11] = val(avp_ring[3]['xyz'][2], scale=FLEXION_SCALE)

        # 엄지 (Thumb)
        # j12는 엄지 맨 밑 로테이션 (이거 없으면 핀치시 엄지가 안쪽으로 로테이션 안됨)
        # j14는 엄지가 일자로 안 펴지는거 피는용도

        avp_thb = angles_dict["Thumb"]

        idx_j12 = get_idx("j12")
        actions[idx_j12] = val(avp_thb[0]['xyz'][1], scale=3.0) 

        idx_j13 = get_idx("j13")
        actions[idx_j13] =  val(avp_thb[1]['xyz'][1])

        idx_j14 = get_idx("j14")
        actions[idx_j14] = val(avp_thb[1]['xyz'][2], scale=3.0) # 엄지 굽힘은 이 값으로 해결

        idx_j15 = get_idx("j15")
        actions[idx_j15] = val(avp_thb[2]['xyz'][2])
        main_end_time = time.perf_counter()
        main_elapsed_time = main_end_time - main_start_time
        # print(f'main 코드 실행 시간 : {main_elapsed_time:.5f} 초\n')
        print('\n')
        return actions
    
    def map_to_right_robot_action(self, angles_dict, dof_names): #오른손용
        actions = np.zeros(len(dof_names))

        def get_idx(name): # dof_names에서 이름 찾아 그 이름의 인덱스 번호 리턴. 조인트 이름에 맞춰서 바꾸면 됨
            try: return dof_names.index(name)
            except ValueError: return None

        def val(deg, scale=1.0):
            return np.deg2rad(deg) * scale
        
        FLEXION_SCALE = 1.4 # 손등을 보고 있을 때 손가락 굽힘이 잘 안되는 것 방지용

        # 검지 (Index)
        avp_idx = angles_dict["Index"]
        
        idx_j0 = get_idx("rj0")
        actions[idx_j0] = val(avp_idx[1]['xyz'][1], scale=0.5)

        idx_j1 = get_idx("rj1")
        actions[idx_j1] = val(avp_idx[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j2 = get_idx("rj2")
        actions[idx_j2] = val(avp_idx[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j3 = get_idx("rj3")
        actions[idx_j3] = val(avp_idx[3]['xyz'][2], scale=FLEXION_SCALE)

        # 중지 (Middle)
        avp_mid = angles_dict["Middle"]
        
        idx_j4 = get_idx("rj4")
        actions[idx_j4] = val(avp_mid[1]['xyz'][1], 0.5)

        idx_j5 = get_idx("rj5")
        actions[idx_j5] = val(avp_mid[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j6 = get_idx("rj6")
        actions[idx_j6] = val(avp_mid[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j7 = get_idx("rj7")
        actions[idx_j7] = val(avp_mid[3]['xyz'][2], scale=FLEXION_SCALE)

        # 약지 (Ring)
        avp_ring = angles_dict["Ring"]
        
        idx_j8 = get_idx("rj8")
        actions[idx_j8] = val(avp_ring[1]['xyz'][1], 0.5)

        idx_j9 = get_idx("rj9")
        actions[idx_j9] = val(avp_ring[1]['xyz'][2], scale=FLEXION_SCALE)

        idx_j10 = get_idx("rj10")
        actions[idx_j10] = val(avp_ring[2]['xyz'][2], scale=FLEXION_SCALE)

        idx_j11 = get_idx("rj11")
        actions[idx_j11] = val(avp_ring[3]['xyz'][2], scale=FLEXION_SCALE)

        # 엄지 (Thumb)
        # rj12는 엄지 맨 밑 로테이션 (이거 없으면 핀치시 엄지가 안쪽으로 로테이션 안됨)
        # rj14는 엄지가 일자로 안 펴지는거 피는용도
        avp_thb = angles_dict["Thumb"]

        idx_j12 = get_idx("rj12")
        actions[idx_j12] = val(avp_thb[0]['xyz'][1], scale=-3.0) 

        idx_j13 = get_idx("rj13")
        actions[idx_j13] =  val(avp_thb[1]['xyz'][1])

        idx_j14 = get_idx("rj14")
        actions[idx_j14] = val(avp_thb[1]['xyz'][2], scale=3.0) # 엄지 굽힘은 이 값으로 해결

        idx_j15 = get_idx("rj15")
        actions[idx_j15] = val(avp_thb[2]['xyz'][2])

        return actions