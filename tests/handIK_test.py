import numpy as np
from scipy.optimize import minimize
from avp_stream import VisionProStreamer
from scipy.spatial.transform import Rotation as R

AVP_IP = "192.168.10.224"  # 사용 환경에 맞게 IP 수정

class LEAPHandIK:
    def __init__(self, finger_type='index'):
        """
        Piper/LEAP Hand 전용 IK Solver
        finger_type: 'index', 'middle', 'ring', 'thumb'
        """
        self.finger_type = finger_type

        # 1. URDF 분석 기반 링크 길이 (단위: m)
        if finger_type == 'thumb':
            # 엄지: Base->MCP(j13), MCP->PIP(j14), PIP->DIP(j15) (j12는 Yaw 담당)
            self.link_lengths = [0.0193, 0.0223, 0.0466] 
            # 엄지 관절 리미트 (Radian)
            self.bounds = [(-0.349, 2.094), (-0.47, 2.443), (-1.20, 1.90), (-1.34, 1.88)]
        else:
            # 검지/중지/소지: MCP(L1), PIP(L2), DIP(L3)
            self.link_lengths = [0.0425, 0.0245, 0.0361]
            # 일반 손가락 관절 리미트 (Radian) [Yaw, MCP, PIP, DIP]
            self.bounds = [(-1.047, 1.047), (-0.314, 2.23), (-0.506, 1.885), (-0.366, 2.042)]

    def forward_kinematics(self, angles):
        L1, L2, L3 = self.link_lengths
        yaw, th1, th2, th3 = angles
        
        # Pitch 평면(XZ) 도달 거리 (엄지는 YZ 평면 등 축이 다를 수 있으나, 로컬 기준 단순화)
        # 일반 손가락: X축이 뻗는 방향, Z축이 굽힘 높이로 가정
        r = L1*np.cos(th1) + L2*np.cos(th1+th2) + L3*np.cos(th1+th2+th3)
        z = L1*np.sin(th1) + L2*np.sin(th1+th2) + L3*np.sin(th1+th2+th3)
        
        return np.array([r*np.cos(yaw), r*np.sin(yaw), z])

    def solve_ik(self, target_pos_local):
        """
        target_pos_local: [x, y, z] (손가락 시작점(MCP) 기준 로컬 좌표)
        """
        # 초기 추정값 (중간 정도 굽힌 상태)
        initial_guess = [0.0, 0.5, 0.5, 0.5]

        def objective(angles):
            current_pos = self.forward_kinematics(angles)
            dist_error = np.linalg.norm(current_pos - target_pos_local)
            
            # 자연스러운 움직임 유도 (PIP-DIP 커플링)
            w_couple = 0.05 if self.finger_type == 'thumb' else 0.1
            coupling_error = (angles[2] - angles[3])**2 * w_couple
            
            return dist_error + coupling_error

        # SLSQP: 제약조건(bounds)이 있는 최적화에 빠르고 효율적
        result = minimize(objective, initial_guess, method='SLSQP', bounds=self.bounds, tol=1e-4)
        return result.x

class IKBasedMapper:
    def __init__(self):
        print("[IK-Mapper] Initializing VisionPro Streamer...")
        self.vps = VisionProStreamer(ip=AVP_IP, record=True)
        
        # 각 손가락별 IK Solver 초기화
        self.solvers = {
            'index': LEAPHandIK('index'),
            'middle': LEAPHandIK('middle'),
            'ring': LEAPHandIK('ring'),
            'thumb': LEAPHandIK('thumb')
        }
        
        # AVP 키포인트 인덱스 매핑 (Tip 위치 인덱스)
        # 0:Wrist, 1-4:Thumb, 5-8:Index, 9-12:Middle, 13-16:Ring, 17-20:Pinky
        # VisionProStreamer의 데이터 구조에 따라 인덱스 확인 필요 (보통 4x4 matrix 리스트로 옴)
        self.TIP_INDICES = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,  # Piper는 4지 로봇이지만 Ring 데이터로 매핑
        }
        
        # 손목 중심에서 각 손가락 MCP까지의 오프셋 (단위: m) - 캘리브레이션 값
        # 이 값을 빼주어야 "손가락 로컬 좌표계"가 됨
        self.MCP_OFFSETS = {
            'index':  np.array([0.03, -0.01, 0.0]),  # 예시 값 (조정 필요)
            'middle': np.array([0.03,  0.00, 0.0]),
            'ring':   np.array([0.03,  0.01, 0.0]),
            'thumb':  np.array([0.01,  0.03, -0.01])
        }
        
        print("[IK-Mapper] Solver Ready.")

    def get_avp_position(self, keypoints, index):
        """4x4 행렬에서 위치(Translation)만 추출"""
        if index >= len(keypoints): return np.zeros(3)
        # keypoints[index]가 4x4 행렬이라고 가정 (avp_stream 구조 확인)
        # 만약 [x, y, z] 리스트라면 그대로 반환
        mat = np.array(keypoints[index])
        if mat.shape == (4, 4):
            return mat[:3, 3]
        elif mat.shape == (3,):
            return mat
        return np.zeros(3)

    def process_frame(self, dof_names, side="left"):
        """
        한 프레임 처리하여 로봇 액션 반환
        side: "left" or "right"
        """
        data = self.vps.latest
        if not data:
            return np.zeros(len(dof_names))

        # 데이터 키 이름 확인 (avp_stream 버전에 따라 다를 수 있음)
        key_hand = f"{side}_joint_poses" if f"{side}_joint_poses" in data else f"{side}_fingers"
        key_wrist = f"{side}_wrist"
        
        hand_transforms = data.get(key_hand, [])
        wrist_transform = data.get(key_wrist, np.eye(4))
        
        if len(hand_transforms) == 0:
            return np.zeros(len(dof_names))

        # 1. Wrist 위치 추출 (World 기준)
        wrist_pos = np.array(wrist_transform)[:3, 3]
        # Wrist Rotation 추출 (좌표계 변환용)
        wrist_rot = np.array(wrist_transform)[:3, :3]
        wrist_rot_inv = wrist_rot.T  # World -> Wrist Local 변환 행렬

        actions = np.zeros(len(dof_names))
        prefix = "rj" if side == "right" else "j"

        def get_idx(name):
            try: return dof_names.index(name)
            except ValueError: return None

        # 2. 각 손가락 별로 IK 계산
        for finger_name, solver in self.solvers.items():
            tip_idx = self.TIP_INDICES[finger_name]
            
            # Tip 위치 (World)
            tip_pos_world = self.get_avp_position(hand_transforms, tip_idx)
            
            # 3. 좌표계 변환: World -> Wrist Local
            # (Tip 위치 - Wrist 위치) 를 손목 회전의 역행렬로 회전
            # 결과: 손목이 원점이고, 손바닥이 기준이 된 좌표
            pos_relative = tip_pos_world - wrist_pos
            pos_local_wrist = wrist_rot_inv @ pos_relative
            
            # 4. 좌표계 변환: Wrist Local -> Finger MCP Local
            # 손목에서 해당 손가락 시작점(MCP)까지의 거리를 뺌
            mcp_offset = self.MCP_OFFSETS[finger_name]
            target_pos_ik = pos_local_wrist - mcp_offset
            
            # 5. 좌표축 재정렬 (AVP -> Piper IK)
            # AVP Local: X(우), Y(상), Z(후) 등일 수 있음 -> Piper IK: X(전방), Y(좌우), Z(상하)
            # 이 부분은 실제 데이터를 보고 축을 섞어야 함 (Mapping)
            # 예시: AVP(Right, Up, Back) -> IK(Forward, Side, Up)
            # IK X = -AVP Z (Forward)
            # IK Y =  AVP X (Side)
            # IK Z =  AVP Y (Up)
            # 아래는 가상의 변환입니다. 실제 움직임을 보며 수정하세요.
            ik_input = np.array([
                -target_pos_ik[2],  # Forward (Depth)
                 target_pos_ik[0],  # Side (Width)
                 target_pos_ik[1]   # Up (Height)
            ])

            # 6. IK 풀기
            angles = solver.solve_ik(ik_input)
            
            # 7. 결과 매핑 (Rad -> Robot Joint)
            # 검지 예시: [Yaw, MCP, PIP, DIP] -> [j0, j1, j2, j3]
            # 주의: Piper는 j1이 MCP Flexion, j0가 Yaw일 수 있음 (URDF 확인 필수)
            
            if finger_name == 'index':
                if get_idx(f"{prefix}0") is not None: actions[get_idx(f"{prefix}0")] = angles[0] # Yaw
                if get_idx(f"{prefix}1") is not None: actions[get_idx(f"{prefix}1")] = angles[1] # MCP
                if get_idx(f"{prefix}2") is not None: actions[get_idx(f"{prefix}2")] = angles[2] # PIP
                if get_idx(f"{prefix}3") is not None: actions[get_idx(f"{prefix}3")] = angles[3] # DIP
                
            elif finger_name == 'middle':
                if get_idx(f"{prefix}4") is not None: actions[get_idx(f"{prefix}4")] = angles[0]
                if get_idx(f"{prefix}5") is not None: actions[get_idx(f"{prefix}5")] = angles[1]
                if get_idx(f"{prefix}6") is not None: actions[get_idx(f"{prefix}6")] = angles[2]
                if get_idx(f"{prefix}7") is not None: actions[get_idx(f"{prefix}7")] = angles[3]
                
            elif finger_name == 'ring':
                if get_idx(f"{prefix}8") is not None: actions[get_idx(f"{prefix}8")] = angles[0]
                if get_idx(f"{prefix}9") is not None: actions[get_idx(f"{prefix}9")] = angles[1]
                if get_idx(f"{prefix}10") is not None: actions[get_idx(f"{prefix}10")] = angles[2]
                if get_idx(f"{prefix}11") is not None: actions[get_idx(f"{prefix}11")] = angles[3]
                
            elif finger_name == 'thumb':
                # 엄지는 Yaw(j12), Base(j13), PIP(j14), DIP(j15)
                if get_idx(f"{prefix}12") is not None: actions[get_idx(f"{prefix}12")] = angles[0]
                if get_idx(f"{prefix}13") is not None: actions[get_idx(f"{prefix}13")] = angles[1]
                if get_idx(f"{prefix}14") is not None: actions[get_idx(f"{prefix}14")] = angles[2]
                if get_idx(f"{prefix}15") is not None: actions[get_idx(f"{prefix}15")] = angles[3]

        return actions