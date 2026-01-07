import xml.etree.ElementTree as ET
import math
import numpy as np

# ---------------------------------------------------------------------------
# 1. 설정: 파일 경로 및 회전 각도 설정
# ---------------------------------------------------------------------------
input_urdf_path = "/home/youngwoo/newTeleop/rmpflow/robot/urdf/dual_arm_robot.urdf"  # ⚠️ 불러올 URDF 경로 수정
output_urdf_path = "/home/youngwoo/newTeleop/rmpflow/robot/urdf/rotated_dual_arm_robot.urdf" # ⚠️ 저장할 URDF 경로 수정


# 회전값 (Y 90 -> Z 90)
rotation_rpy_str = "0 0 3.14" 

def rotate_and_save_urdf_with_inertia(in_path, out_path, rpy_str):
    try:
        tree = ET.parse(in_path)
        robot_root = tree.getroot()
    except Exception as e:
        print(f"[❌] Error loading URDF: {e}")
        return

    # 1. 현재 Root Link 찾기
    links = [link.get('name') for link in robot_root.findall('link')]
    joints = robot_root.findall('joint')
    child_links = [joint.find('child').get('link') for joint in joints]
    
    root_link_name = None
    for link in links:
        if link not in child_links:
            root_link_name = link
            break
            
    if not root_link_name:
        print("[❌] Could not identify the root link.")
        return

    print(f"[ℹ️] Detected Root Link: {root_link_name}")

    # 2. 새로운 더미 Root Link 생성 ('world_rotation_fix')
    new_root_name = "world_rotation_fix"
    new_link = ET.Element('link', {'name': new_root_name})
    
    # ⭐ [추가된 부분] Inertial 태그 추가 (경고 제거용) ⭐
    # 질량은 매우 작게(0.001kg) 설정하여 물리적 영향을 최소화
    inertial = ET.SubElement(new_link, 'inertial')
    ET.SubElement(inertial, 'mass', {'value': '1e-5'}) # 거의 0에 가까운 질량
    # 관성 텐서 (대각선 성분만 작게 설정)
    ET.SubElement(inertial, 'inertia', {
        'ixx': '1e-6', 'ixy': '0', 'ixz': '0',
        'iyy': '1e-6', 'iyz': '0',
        'izz': '1e-6'
    })
    
    robot_root.insert(0, new_link)

    # 3. Joint 생성
    new_joint = ET.Element('joint', {'name': 'fixed_world_rotation', 'type': 'fixed'})
    ET.SubElement(new_joint, 'parent', {'link': new_root_name})
    ET.SubElement(new_joint, 'child', {'link': root_link_name})
    ET.SubElement(new_joint, 'origin', {'xyz': '0 0 0', 'rpy': rpy_str})
    
    robot_root.append(new_joint)

    # 4. 저장
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"[✅] Fixed URDF saved to: {out_path}")

# 실행
rotate_and_save_urdf_with_inertia(input_urdf_path, output_urdf_path, rotation_rpy_str)