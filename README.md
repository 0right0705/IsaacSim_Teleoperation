# IsaacSim_Teleoperation
Bimanual Teleoperation System using RMPflow and XR Hand Tracking

## 폴더별 특징
1. **webxr** : HMD(Vision Pro / Meta Quest)에 Isaac Sim 환경 스트리밍 없이 Teleoperation만 진행
2. **webxr_cameraView** : HMD에 Isaac Sim 환경 스트리밍 및 고정 Camera Perspective
3. **webxr_movingCamera** : HMD에 Isaac Sim 환경 스트리밍 및 Head 좌표에 따른 Camera 좌표 이동
4. **webxr_movingRobot** : HMD에 Isaac Sim 환경 스트리밍 및 Head 좌표에 따른 Camera 좌표 및 로봇 좌표 이동
5. **grabStrawberries** : HMD에 Isaac SIm 환경 스트리밍 및 Head 좌표에 따른 Camera 좌표 이동 및 특정 mesh(Test는 연구실에서 만든 Strawberry USD로 사용하였으나 제공 X, 직접 mesh 생성 후 테스트 필요) grasping 테스트


## 사용 방법
### Server Computer에서 - 
1. $ cd {원하는 폴더 이름} # 예시 : cd grabStrawberries
2. $ ifconfig # 본인의 IP 주소 확인 후 handview.html, mainHandTeleop.py, streamer.py 파일에서 IP 주소 변경 (포트 번호는 그대로)
3. $ openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365  // cert.pem, key.pem 생성(https 통신용)
4. $ python https_server.py
5. $ python main.py # Isaac Sim이 설치되어 있어야 함

### Meta Quest3 내의 브라우저 접속 후 - 
1. https://{본인 IP 주소}:4443/handview.html 
2. 브라우저 하단의 Start XR 버튼 클릭
3. Hand Calibration 후 Teleoperation 진행

### Vision Pro 내의 브라우저 접속 후 - 
1. https://{본인 IP 주소}:8765 접속 후 "Your connection is not private" 뜨면 proceed to ... (unsafe) 버튼 클릭, 에러가 떠있어도 상관 없음. 
2. 새 tab에서 https://{본인 IP 주소}:4443/handview.html 접속
3. https://{본인 IP 주소}:4443/handview.html에서 노란색 배경만 뜬다면, 하단의 Start XR 누른 후 핀치로 빠져나온 후 Isaac Sim 환경이 뜰 때까지 위 1번, 2번 반복.
4. 브라우저 하단의 Start XR 버튼 클릭
5. Hand Calibration 후 Teleoperation 진행

## 이 프로젝트는 Openarm & Tesollo DG5F USD로 진행함.
> 다른 로봇 사용 시 
1. usd 파일 주소 변경
2. 해당 로봇 팔에 맞는 yaml 파일 작성
3. USD 파일에서 Hand Joint 이름을 찾아 mainHandTeleop.py의 get_joint_index()에 넣기
4. 사용 Hand의 손가락 관절 매핑하기