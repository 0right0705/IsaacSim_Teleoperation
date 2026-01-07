import omni.usd
# Usd, UsdGeom, Gf는 그대로 두고, UsdPhysics를 추가로 임포트합니다.
from pxr import UsdGeom, Gf, Usd, UsdPhysics 

stage = omni.usd.get_context().get_stage()
prim_path = "/World/openarm_tesollo_mount"
# 저장 경로 변수명을 다시 Z-축 변수명 대신 X-축 변수명으로 맞춥니다.
NEW_USD_PATH = "/home/youngwoo/rotated_openarm_tesollo.usd"


prim = stage.GetPrimAtPath(prim_path)
if not prim.IsValid():
    print(f"[❌] Prim not found at {prim_path}. 로봇이 스테이지에 로드되어 있는지 확인하세요.")
else:
    # 1. 기존 변환 적용 (Y: 90도, X: -90도)
    xform = UsdGeom.Xformable(prim)

    # Y축 90도 회전
    # y_rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
    # y_rot_matrix = Gf.Matrix4d() 
    # y_rot_matrix.SetRotate(y_rotation)
    
    # z축 -90도 회전
    z_rotation = Gf.Rotation(Gf.Vec3d(0, 0, 0), 0) 
    z_rot_matrix = Gf.Matrix4d()
    z_rot_matrix.SetRotate(z_rotation)

    # 최종 행렬: X(-90) * Y(90)
    final_matrix = z_rot_matrix 
    
    xform_op = xform.GetTransformOp()
    if not xform_op:
        xform_op = xform.AddTransformOp()
    
    xform_op.Set(final_matrix)
    print(f"[✅] Applied combined Y (90°) and X (-90°) rotation to {prim_path}.")

    # -------------------------------------------------------------------
    # 2. ArticulationRootAPI 적용 (⭐ ArticulationRootAPI 모듈 경로 수정)
    # -------------------------------------------------------------------
    
    # Usd.ModelAPI 적용 (이전 단계에서 수정한 생성자 방식 유지)
    if not prim.HasAPI(Usd.ModelAPI):
        Usd.ModelAPI(prim) 
        print(f"[✅] Applied ModelAPI to {prim_path}.")
    
    # ArticulationRootAPI 적용: UsdPhysics 모듈 사용
    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI): # 💡 UsdPhysics.ArticulationRootAPI 로 수정
        UsdPhysics.ArticulationRootAPI(prim)            # 💡 UsdPhysics.ArticulationRootAPI(prim) 로 수정
        print(f"[✅] Applied ArticulationRootAPI to {prim_path}.")

    # # -------------------------------------------------------------------
    # # 3. 새로운 USD 파일로 저장
    # # -------------------------------------------------------------------
    # try:
    #     stage.GetRootLayer().Export(NEW_USD_PATH)
    #     print(f"[💾] Successfully saved modified USD to: {NEW_USD_PATH}")
    # except Exception as e:
    #     print(f"[❌] Failed to save USD: {e}")import omni.usd
# Usd, UsdGeom, Gf는 그대로 두고, UsdPhysics를 추가로 임포트합니다.
from pxr import UsdGeom, Gf, Usd, UsdPhysics 

stage = omni.usd.get_context().get_stage()


prim = stage.GetPrimAtPath(prim_path)
if not prim.IsValid():
    print(f"[❌] Prim not found at {prim_path}. 로봇이 스테이지에 로드되어 있는지 확인하세요.")
else:
    # 1. 기존 변환 적용 (Y: 90도, X: -90도)
    xform = UsdGeom.Xformable(prim)

    # Y축 90도 회전
    # y_rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
    # y_rot_matrix = Gf.Matrix4d() 
    # y_rot_matrix.SetRotate(y_rotation)
    
    # z축 -90도 회전
    z_rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90.0) 
    z_rot_matrix = Gf.Matrix4d()
    z_rot_matrix.SetRotate(z_rotation)

    # 최종 행렬: X(-90) * Y(90)
    final_matrix = z_rot_matrix 
    
    xform_op = xform.GetTransformOp()
    if not xform_op:
        xform_op = xform.AddTransformOp()
    
    xform_op.Set(final_matrix)
    print(f"[✅] Applied combined Y (90°) and X (-90°) rotation to {prim_path}.")

    # -------------------------------------------------------------------
    # 2. ArticulationRootAPI 적용 (⭐ ArticulationRootAPI 모듈 경로 수정)
    # -------------------------------------------------------------------
    
    # Usd.ModelAPI 적용 (이전 단계에서 수정한 생성자 방식 유지)
    if not prim.HasAPI(Usd.ModelAPI):
        Usd.ModelAPI(prim) 
        print(f"[✅] Applied ModelAPI to {prim_path}.")
    
    # ArticulationRootAPI 적용: UsdPhysics 모듈 사용
    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI): # 💡 UsdPhysics.ArticulationRootAPI 로 수정
        UsdPhysics.ArticulationRootAPI(prim)            # 💡 UsdPhysics.ArticulationRootAPI(prim) 로 수정
        print(f"[✅] Applied ArticulationRootAPI to {prim_path}.")

    # # # -------------------------------------------------------------------
    # # # 3. 새로운 USD 파일로 저장
    # # # -------------------------------------------------------------------
    # try:
    #     stage.GetRootLayer().Export(NEW_USD_PATH)
    #     print(f"[💾] Successfully saved modified USD to: {NEW_USD_PATH}")
    # except Exception as e:
    #     print(f"[❌] Failed to save USD: {e}")
