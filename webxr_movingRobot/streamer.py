import asyncio
import websockets
import json
import ssl
import threading
from pathlib import Path

class Streamer:
    def __init__(
        self,
        ip="192.168.10.23",
        port=8765,
        max_queue_size=1,
        use_ssl=True,
        certfile="cert.pem",
        keyfile="key.pem",
    ):
        self.ip = ip
        self.port = port
        self.use_ssl = use_ssl
        self.certfile = Path(__file__).resolve().parent / certfile
        self.keyfile = Path(__file__).resolve().parent / keyfile
        # maxsize=1로 설정하여 가장 최신 프레임만 유지하는 것을 권장 (지연 방지)
        self.raw_data_queue = asyncio.Queue(maxsize=max_queue_size)
        self._loop = None
        self._thread = None
        self._server_task = None
        self.connected_clients = set() # 접속된 클라이언트를 담을 세트 추가
        self._loop = None # 내부에서 사용하는 루프 저장용

    async def handle_connection(self, websocket): # 기존 함수 시작
        """메타퀘스트로부터 데이터를 받아 큐에 넣는 핸들러"""
        # 클라이언트 접속 시 목록에 추가
        self.connected_clients.add(websocket) 
        print(f"Client Connected: {websocket.remote_address} | Total: {len(self.connected_clients)}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if self.raw_data_queue.full():
                        try:
                            self.raw_data_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.raw_data_queue.put(data)
                except json.JSONDecodeError:
                    print("JSON 디코딩 오류")
                except Exception as e:
                    print(f"데이터 처리 중 오류: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Client Disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"연결 오류: {e}")
        finally:
            # [수정] 연결 종료 시 목록에서 확실히 제거
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
                print(f"Client removed from broadcast list. Remaining: {len(self.connected_clients)}")

    async def start_server(self):
        """웹소켓 서버 시작"""
        ssl_context = None
        if self.use_ssl:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
        
        print(f"Starting server on {self.ip}:{self.port}...")
        async with websockets.serve(self.handle_connection, self.ip, self.port, ssl=ssl_context):
            await asyncio.Future()  # 서버가 종료되지 않고 계속 실행되도록 함

    async def get_latest_data(self):
        """다른 로직에서 데이터를 꺼내갈 때 사용하는 메서드"""
        return await self.raw_data_queue.get()

    def start_background_server(self):
        """별도 쓰레드에서 asyncio 이벤트 루프로 서버 실행 (동기 코드용)"""
        if self._thread and self._thread.is_alive():
            return

        self._loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(self._loop)
            self._server_task = self._loop.create_task(self.start_server())
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def get_latest_data_sync(self, timeout=None):
        """동기 코드에서 최신 데이터를 blocking 방식으로 읽기"""
        if not self._loop:
            raise RuntimeError("Server loop not running. Call start_background_server() first.")
        future = asyncio.run_coroutine_threadsafe(self.get_latest_data(), self._loop)
        return future.result(timeout=timeout)
    
    def get_latest_nowait(self):
        """큐에서 데이터를 기다리지 않고 즉시 확인 (데이터 없으면 None 반환)"""
        try:
            return self.raw_data_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_latest_data_sync_nowait(self):
        """동기 코드에서 호출 가능한 비차단 데이터 취득 메서드"""
        if not self._loop:
            return None
        future = asyncio.run_coroutine_threadsafe(asyncio.to_thread(self.get_latest_nowait), self._loop)
        try:
            # 매우 짧은 타임아웃을 주거나 바로 결과 확인
            return future.result(timeout=0.001)
        except:
            return None
    # streamer.py 내부 Streamer 클래스에 추가
    def broadcast(self, message):
        """연결된 모든 VR 기기로 데이터를 보냅니다."""
        if not self.connected_clients:
            return

        # 비동기 루프를 통해 모든 클라이언트에게 전송
        for client in list(self.connected_clients):
            try:
                # 퀘스트/비전프로 브라우저는 비동기로 메시지를 보내야 합니다.
                asyncio.run_coroutine_threadsafe(client.send(message), self._loop)
            except Exception as e:
                print(f"[Streamer] Send error: {e}")

# --- 실행 예시 (main) ---
async def main():
    streamer = Streamer(ip="192.168.10.23", port=8765)
    server_task = asyncio.create_task(streamer.start_server())
    print("데이터 수신 대기 중...")
    try:
        while True:
            # 큐에서 데이터 꺼내기
            data = await streamer.get_latest_data()
            print(data)
            print('--' * 10)
            
    except asyncio.CancelledError:
        server_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())