# IsaacSim_Teleoperation
Bimanual Teleoperation System using RMPflow and XR Hand Tracking

## 폴더별 특징
1. **`webxr`** : HMD(Vision Pro / Meta Quest)에 Isaac Sim 환경 스트리밍 없이 Teleoperation만 진행
2. **`webxr_cameraView`** : HMD에 Isaac Sim 환경 스트리밍 및 고정 Camera Perspective
3. **`webxr_movingCamera`** : HMD에 Isaac Sim 환경 스트리밍 및 Head 좌표에 따른 Camera 좌표 이동
4. **`webxr_movingRobot`** : HMD에 Isaac Sim 환경 스트리밍 및 Head 좌표에 따른 Camera 좌표 및 로봇 좌표 이동
5. **`grabStrawberries`** : HMD에 Isaac SIm 환경 스트리밍 및 Head 좌표에 따른 Camera 좌표 이동 및 특정 mesh(Test는 연구실에서 만든 Strawberry USD로 사용하였으나 제공 X, 직접 mesh 생성 후 테스트 필요) grasping 테스트


## 사용 방법
### Server Computer에서 - 
1. $ **`cd {원하는 폴더 이름}`** # 예시 : cd grabStrawberries
2. $ **`ifconfig`** # 본인의 IP 주소 확인 후 handview.html, mainHandTeleop.py, streamer.py 파일에서 IP 주소 변경 (포트 번호는 그대로)
3. $ **`openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365`**  // cert.pem, key.pem 생성(https 통신용)
4. $ **`python https_server.py`**
5. $ **`python main.py`** # Isaac Sim이 설치되어 있어야 함

### Meta Quest3 내의 브라우저 접속 후 - 
1. **`https://{본인 IP 주소}:4443/handview.html`**
2. 브라우저 하단의 Start XR 버튼 클릭
3. Hand Calibration 후 Teleoperation 진행

### Vision Pro 내의 브라우저 접속 후 - 
1. **`https://{본인 IP 주소}:8765`** 접속 후 "Your connection is not private" 뜨면 proceed to ... (unsafe) 버튼 클릭, 에러가 떠있어도 상관 없음. 
2. 새 tab에서 **`https://{본인 IP 주소}:4443/handview.html`** 접속
3. **`https://{본인 IP 주소}:4443/handview.html`** 에서 노란색 배경만 뜬다면, 하단의 Start XR 누른 후 핀치로 빠져나온 후 Isaac Sim 환경이 뜰 때까지 위 1번, 2번 반복.
4. 브라우저 하단의 Start XR 버튼 클릭
5. Hand Calibration 후 Teleoperation 진행

## 이 프로젝트는 Openarm & Tesollo DG5F USD로 진행함.
> 다른 로봇 사용 시 
1. usd 파일 주소 변경
2. 해당 로봇 팔에 맞는 yaml 파일 작성
3. USD 파일에서 Hand Joint 이름을 찾아 mainHandTeleop.py의 get_joint_index()에 넣기
4. 사용 Hand의 손가락 관절 매핑하기

***

## English Description


## Folder Features
1. **`webxr`**: Teleoperation only without streaming the Isaac Sim environment to the HMD.
2. **`webxr_cameraView`**: Isaac Sim environment streaming with a **Fixed Camera Perspective**.
3. **`webxr_movingCamera`**: Isaac Sim streaming with **Camera movement** synchronized with the HMD Head coordinates.
4. **`webxr_movingRobot`**: Isaac Sim streaming with both **Camera and Robot Base movement** synchronized with the Head coordinates.
5. **`grabStrawberries`**: Grasping test with moving camera based on Head tracking. 
   - *Note: The Strawberry USD created in our lab is not provided. Please create your own mesh for testing.*

## How to Use
### On the Server Computer - 
1. Navigate to the project directory: **`cd {folder_name}`**.
2. Run **`ifconfig`** to find your IP address. Update the IP address field in **`handview.html`**, **`mainHandTeleop.py`**, and **`streamer.py`** (do not change the port numbers).
3. Generate SSL certificates for secure HTTPS communication using the OpenSSL command **`openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365`**
4. Start the HTTPS server by running **`python https_server.py`**.
5. Execute the main script by running **`python main.py`**.. Note: This requires NVIDIA Isaac Sim to be installed on your system.

## Accessing via Meta Quest 3 Browser
1. Navigate to: **`https://{Your_IP_Address}:4443/handview.html`**
2. Click the Start XR button at the bottom of the browser.
3. Complete Hand Calibration, then proceed with Teleoperation.

## Accessing via Apple Vision Pro Browser
1. Navigate to **`https://{Your_IP_Address}:8765`**. If a "Your connection is not private" warning appears, click "Advanced" and then "Proceed to... (unsafe)". (It is okay if an error is displayed on this page).
2. Open a new tab and navigate to **`https://{Your_IP_Address}:4443/handview.html`**.
3. If only a yellow background appears, click Start XR, exit the view using a pinch gesture, and repeat steps 1 and 2 until the Isaac Sim environment is visible.
4. Click the Start XR button at the bottom of the browser.
5. Complete Hand Calibration, then proceed with Teleoperation.

## Project Configuration
This project is configured for Openarm & Tesollo DG5F USD.

> To use a different robot:

1. Update the USD file path.
2. Create a YAML configuration file specific to the robot arm.
3. Locate the Hand Joint names in the USD file and update the get_joint_index() function in mainHandTeleop.py.
4. Map the finger joints for the hand hardware you are using.