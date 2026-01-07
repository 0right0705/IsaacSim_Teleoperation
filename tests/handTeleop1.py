import numpy as np
from avp_stream import VisionProStreamer
from scipy.spatial.transform import Rotation as R

AVP_IP = "10.102.101.196"


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
        """AVP 좌표 → Isaac 좌표 변환"""
        return (self.R_avp_to_isaac @ p.reshape(3, )).reshape(3, )

    def get_avp_data(self):
        # 예외 처리 복구: 데이터가 없거나 딕셔너리가 비어있을 때 터지는 것을 방지
        try:
            r = self.vps.latest
            if r is None or "left_fingers" not in r:
                return None
            
            left = np.array(r["left_fingers"])
            
            # 차원 문제 방지 (가끔 (1, 25, 4, 4)로 들어올 때가 있음)
            if left.ndim == 4: 
                left = left[0]
                
            return left

        except Exception as e:
            # 에러가 나도 죽지 않고 None을 반환하여 다음 루프를 돌게 함
            print(f"[Warning] get_avp_data error: {e}")
            return None
        # 데이터가 없는 경우 처리
        # if r is None or "left_fingers" not in r:
        #     return None, None

        # try:
        #     left = np.array(r["left_fingers"])
        #     right = np.array(r["right_fingers"])

        #     # # === [수정된 부분] ===
        #     # # (25, 4, 4) 형상이면 그대로 둡니다. 
        #     # # 만약 (Batch, 25, 4, 4) 처럼 4차원으로 들어올 때만 압축을 풉니다.
        #     # if l.ndim == 4: l = l[0]
        #     # if r_.ndim == 4: r_ = r_[0]
        #     # # ===================

        #     # # 디버깅용 (필요시 주석 해제)
        #     # # print(f"Processing shape - Left: {l.shape}, Right: {r_.shape}")

        #     # left_q = self.compute_hand_angles(l)
        #     # right_q = self.compute_hand_angles(r_)

        #     # # 최초 1회 neutral offset 저장
        #     # if not self.offset_applied:
        #     #     self.left_offset = -left_q
        #     #     self.right_offset = -right_q
        #     #     self.offset_applied = True
        #     #     print("[Mapping] Neutral calibration captured.")

        #     # left_q = left_q + self.left_offset
        #     # right_q = right_q + self.right_offset

        #     return left, right

        # except Exception as e:
        #     print("[Error] get_avp_data:", e)
        #     # 에러 발생 시 형상 정보 출력 (디버깅 용이)
        #     if 'l' in locals(): print(f"Left Shape was: {l.shape}")
        #     return None, None

    # 손가락 각도 계산 --------------------------------------------

    def calculate_hand_angles(self, transforms):
        # 1번 데이터(인덱스 0)를 손목(Wrist)으로 가정
        WRIST = 0 
        
        # [수정됨] 이미지 기반의 정확한 체인 정의
        # 팁(Tip) 부분(예: 4, 9, 14, 19)은 회전각 계산에 자식이 없으므로 제외해도 됩니다.
        chains = {
            # 엄지: Wrist -> Knuckle(1) -> InterBase(2) -> InterTip(3)
            "Thumb (엄지)": [WRIST, 1, 2, 3],       
            
            # 검지: Wrist -> Metacarpal(5) -> Knuckle(6) -> InterBase(7) -> InterTip(8)
            "Index (검지)": [WRIST, 5, 6, 7, 8],    
            
            # 중지: Wrist -> Metacarpal(10) -> Knuckle(11) -> InterBase(12) -> InterTip(13)
            "Middle (중지)": [WRIST, 10, 11, 12, 13], 
            
            # 약지: Wrist -> Metacarpal(15) -> Knuckle(16) -> InterBase(17) -> InterTip(18)
            "Ring (약지)":   [WRIST, 15, 16, 17, 18]  
        }
        
        results = {}
        
        for name, indices in chains.items():
            angles = []
            for i in range(len(indices) - 1):
                p_idx = indices[i]
                c_idx = indices[i+1]
                
                if c_idx >= len(transforms): continue
                
                # 회전 행렬 추출
                R_parent = transforms[p_idx][:3, :3]
                R_child = transforms[c_idx][:3, :3]
                
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
    def map_to_robot_action(self, angles_dict, dof_names):
        actions = np.zeros(len(dof_names))

        def get_idx(name):
            try: return dof_names.index(name)
            except ValueError: return None

        def val(deg, scale=1.0):
            # 방향이 반대면 -np.deg2rad(deg) 로 수정
            return np.deg2rad(deg) * scale

        try:
            # =========================
            # 1. 검지 (Index)
            # Chain: 0->5(Meta), 5->6(MCP), 6->7(PIP), 7->8(DIP)
            # =========================
            avp_idx = angles_dict["Index (검지)"]
            
            # j0: Abduction (좌우 벌림) -> MCP 관절(5->6)의 Y축 or X축
            idx_j0 = get_idx("j0")
            if idx_j0 is not None:
                actions[idx_j0] = val(avp_idx[1]['xyz'][1], scale=0.5) 

            # j1: MCP Flexion (뿌리 굽힘) -> MCP 관절(5->6)의 Z축
            idx_j1 = get_idx("j1")
            if idx_j1 is not None:
                actions[idx_j1] = val(avp_idx[1]['xyz'][2]) # avp_idx[0] 아님 주의!

            # j2: PIP Flexion -> (6->7)
            idx_j2 = get_idx("j2")
            if idx_j2 is not None:
                actions[idx_j2] = val(avp_idx[2]['xyz'][2])

            # j3: DIP Flexion -> (7->8)
            idx_j3 = get_idx("j3")
            if idx_j3 is not None:
                actions[idx_j3] = val(avp_idx[3]['xyz'][2])

            # =========================
            # 2. 중지 (Middle)
            # Chain: 0->10(Meta), 10->11(MCP)...
            # =========================
            avp_mid = angles_dict["Middle (중지)"]
            
            idx_j4 = get_idx("j4") 
            if idx_j4 is not None: actions[idx_j4] = val(avp_mid[1]['xyz'][1], 0.5)

            idx_j5 = get_idx("j5") # MCP
            if idx_j5 is not None: actions[idx_j5] = val(avp_mid[1]['xyz'][2])

            idx_j6 = get_idx("j6") # PIP
            if idx_j6 is not None: actions[idx_j6] = val(avp_mid[2]['xyz'][2])

            idx_j7 = get_idx("j7") # DIP
            if idx_j7 is not None: actions[idx_j7] = val(avp_mid[3]['xyz'][2])

            # =========================
            # 3. 약지 (Ring)
            # Chain: 0->15(Meta), 15->16(MCP)...
            # =========================
            avp_ring = angles_dict["Ring (약지)"]
            
            idx_j8 = get_idx("j8") 
            if idx_j8 is not None: actions[idx_j8] = val(avp_ring[1]['xyz'][1], 0.5)

            idx_j9 = get_idx("j9") 
            if idx_j9 is not None: actions[idx_j9] = val(avp_ring[1]['xyz'][2])

            idx_j10 = get_idx("j10")
            if idx_j10 is not None: actions[idx_j10] = val(avp_ring[2]['xyz'][2])

            idx_j11 = get_idx("j11")
            if idx_j11 is not None: actions[idx_j11] = val(avp_ring[3]['xyz'][2])

            # =========================
            # 4. 엄지 (Thumb)
            # Chain: 0->1(Knuckle), 1->2(InterBase), 2->3(InterTip)
            # 엄지는 Metacarpal이 따로 인덱싱되지 않고 1번이 바로 Knuckle로 시작하므로 인덱스 유지
            # =========================
            avp_thb = angles_dict["Thumb (엄지)"]
            
            idx_j12 = get_idx("j12") # Abd/Rot
            if idx_j12 is not None: 
                # 엄지 0->1 구간(CMC Joint)의 회전을 사용
                actions[idx_j12] = val(avp_thb[0]['xyz'][1]) 

            idx_j13 = get_idx("j13") # MCP
            if idx_j13 is not None: actions[idx_j13] = val(avp_thb[0]['xyz'][2]) 

            idx_j14 = get_idx("j14") # PIP
            if idx_j14 is not None: actions[idx_j14] = val(avp_thb[1]['xyz'][2])

            idx_j15 = get_idx("j15") # DIP
            if idx_j15 is not None: actions[idx_j15] = val(avp_thb[2]['xyz'][2])

        except Exception as e:
            print(f"[Mapping Error] {e}")

        return actions