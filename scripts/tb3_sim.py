import os
import sys
import time
from queue import Queue
import math

import mujoco as mj
import cv2
import numpy as np

# 프로젝트 루트에서 utils 가져오기
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.mujoco_renderer import MuJoCoViewer
from utils.object_detector import ObjectDetector

ACTION_TABLE = {
    "멈춤": (0.0, 0.0),
    "직진": (8.0, 8.0),
    "후진": (-8.0, -8.0),
    "좌회전": (6.0, 8.0),
    "우회전": (8.0, 6.0),
    "제자리 회전": (4.0, -4.0),
}

class TurtlebotFactorySim:
    """
    MuJoCo 기반 터틀봇3 팩토리 시뮬 통합 클래스.

    기능:
    - tb3_factory_cards.xml 로드
    - 메인뷰 + 로봇 카메라 렌더링
    - latest_frame 에 로봇 카메라 마지막 프레임(BGR) 저장
    - (옵션) YOLO로 로봇 카메라 프레임 감지 & cv2 창으로 출력
    - (옵션) command_queue 에서 명령을 읽어와 apply_command()로 처리
    """

    #wheel encoder를 이용해 회전. 빠르지만 실제 몸체의 회전과 오차가 있음.
    def rotate_by_degrees_fast(self, deg, wheel_radius=0.033, wheel_base=0.16, speed=6.0):
        target = math.radians(deg)  #degree radian 변환해서 최종 회전해야되는 rad 입력받음
        l0, r0, _, _ = self.get_wheel_encoder() #처음 바퀴의 회전량을 구함

        direction = 1.0 if target > 0 else -1.0 #+양수:좌회전, 음수:우회전 -> 로봇을 조금 회전시킴
        self.data.ctrl[0] = -direction * speed
        self.data.ctrl[1] = +direction * speed

        while True: #계속 조금씩 회전시키면서 목표 rad에 도달하면 멈춤.
            self.step_simulation()
            self.render()
            l, r, _, _ = self.get_wheel_encoder()
            dyaw = wheel_radius * ((r - r0) - (l - l0)) / wheel_base    #바퀴의 회전각을 이용해 몸체의 회전량을 구함.

            if (target > 0 and dyaw >= target) or (target < 0 and dyaw <= target):
                break

        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0

    #wheel encoder와 환경의 물리적 법칙까지 반영된 몸체의 회전량을 구함. 느리지만 더 정확함.
    def rotate_by_degrees(self, deg, body_name="base", speed_max=6.0, speed_min=1.5, kp=3.0, tol_deg=0.5):
        target = math.radians(deg)
        yaw0 = self._get_body_yaw(body_name)
        tol = math.radians(tol_deg) #허용 오차

        while True:
            self.step_simulation()
            self.render()

            yaw = self._get_body_yaw(body_name)
            dyaw = self._wrap_pi(yaw - yaw0)    #현재 회전량 - 시작할때 회전량
            err = self._wrap_pi(target - dyaw)  #목표까지의 오차

            if abs(err) <= tol: #목표치까지 달성 -> break
                break
            
            #오차가 클수록 더 빨리 회전하게함
            v = kp * err
            v = max(-speed_max, min(speed_max, v))
            if abs(v) < speed_min:
                v = math.copysign(speed_min, v)

            self.data.ctrl[0] = -v
            self.data.ctrl[1] = +v

        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0

    def _get_body_yaw(self, body_name="base"):  #mujoco body의 정보를 받아와서 그걸로 몸체의 회전량을 구하는 함수.
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise RuntimeError(f"Body '{body_name}' not found")

        w, x, y, z = self.data.xquat[bid]

        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _wrap_pi(self, a):  #계산결과를 가장 작은 각도차이만 나오게함
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def __init__(
        self,
        xml_path: str | None = None,
        use_yolo: bool = False,
        yolo_weight_path: str | None = None,
        yolo_conf: float = 0.5,
        command_queue: Queue | None = None,
        fps: int = 60,
        current_action = None,
        action_end_sim_time = 0.0,
    ):
        # ===== 경로 설정 =====
        script_path = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(scripts_dir)  # /data/jinsup/js_mujoco

        if xml_path is None:
            xml_path = os.path.join(
                project_root,
                "asset",
                "robotis_tb3",
                "tb3_factory_cards.xml",
            )

        print(f"[TurtlebotFactorySim] Loading scene from: {xml_path}")

        # 검색 모드 타겟 레이블
        self.search_target_label = None  
        self.current_action = current_action
        self.action_end_sim_time = action_end_sim_time
        # ===== MuJoCo 모델/데이터 로드 =====
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # 기존 MuJoCoViewer 사용
        self.viewer = MuJoCoViewer(self.model, self.data)

        # ===== 카메라 프레임 저장용 =====
        # 항상 "로봇 카메라 기준 BGR 이미지"를 최신 상태로 보관
        self.latest_frame: np.ndarray | None = None

        # ===== YOLO 옵션 =====
        self.use_yolo = use_yolo
        self.detector = None
        self.yolo_window_name = "Robot YOLO View"

        if self.use_yolo:
            if yolo_weight_path is None:
                raise ValueError("use_yolo=True 인데 yolo_weight_path 가 없습니다.")
            if not os.path.exists(yolo_weight_path):
                raise FileNotFoundError(f"YOLO weight not found: {yolo_weight_path}")

            print(f"[TurtlebotFactorySim] Loading ObjectDetector: {yolo_weight_path}")
            self.detector = ObjectDetector(yolo_weight_path, conf=yolo_conf)

            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.yolo_window_name, 640, 480)

        # ===== 명령 큐 (LLM / 키보드 등에서 넣어주는 명령) =====
        self.command_queue = command_queue if command_queue is not None else Queue()

        # ===== 루프 설정 =====
        self.fps = fps
        self._running = False

        # ===== Continuous drive / Auto avoid =====
        self.cruise_forward = False
        self.cruise_speed = 10.0
        self.avoid_threshold_m = 1.0

        self.auto_scan_mode = False
        self.scan_stage = 0
        self.scan_d_right = None
        self.scan_d_left = None
        self.scan_eps = 0.05

        # ===== Rangefinder sensor 캐싱 =====
        self.rf_front_sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "rf_front")
        if self.rf_front_sid < 0:
            raise RuntimeError(
                "Sensor 'rf_front' not found."
            )
        self.rf_front_adr = int(self.model.sensor_adr[self.rf_front_sid])

        # ===== Wheel encoder sensor 캐싱 =====
        self.enc_l_pos_adr = self._cache_sensor("enc_left_pos")
        self.enc_r_pos_adr = self._cache_sensor("enc_right_pos")
        self.enc_l_vel_adr = self._cache_sensor("enc_left_vel")
        self.enc_r_vel_adr = self._cache_sensor("enc_right_vel")

    # ------------------------------------------------------------------
    # 외부에서 사용할 수 있는 유틸 메서드들
    # ------------------------------------------------------------------
    def _cache_sensor(self, name: str) -> int:
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"Sensor '{name}' not found")
        return int(self.model.sensor_adr[sid])

    def step_simulation(self):
        """한 타임스텝(fps 기준)만큼 시뮬레이션을 진행."""
        time_prev = self.data.time
        dt = 1.0 / self.fps
        while self.data.time - time_prev < dt:
            self.viewer.step_simulation()

    def render(self):
        """메인뷰 + 로봇 카메라 렌더링, latest_frame 업데이트."""
        # 메인 뷰: IMU overlay
        self.viewer.render_main(overlay_type="imu")

        # 로봇 카메라 화면 표시 + 이미지 캡처
        self.viewer.render_robot()
        # MuJoCoViewer 안에 capture_img() 가 로봇 카메라 뷰를 BGR로 반환한다고 가정
        if hasattr(self.viewer, "capture_img"):
            frame_bgr = self.viewer.capture_img()
            self.latest_frame = frame_bgr
        else:
            self.latest_frame = None

        self.viewer.poll_events()

    def apply_command(self, cmd: str, base_duration: float = 1.0):
        cmd = cmd.strip()

        # 연속 직진 모드 ON
        if cmd in ["직진_계속", "FORWARD_CONTINUOUS"]:
            self.cruise_forward = True
            self.auto_scan_mode = False
            self.current_action = None
            self.action_end_sim_time = float("inf")
            self.data.ctrl[0] = self.cruise_speed
            self.data.ctrl[1] = self.cruise_speed
            print("[TurtlebotFactorySim] Continuous forward ON.")
            return

        # 연속 직진 모드 OFF (정지)
        if cmd in ["정지", "STOP", "멈춤"]:
            self.cruise_forward = False
            self.auto_scan_mode = False
            self.current_action = None
            self.action_end_sim_time = 0.0
            self.data.ctrl[0] = 0.0
            self.data.ctrl[1] = 0.0
            print("[TurtlebotFactorySim] STOP.")
            return

        if cmd.startswith("TURN_RIGHT_"):
            deg = float(cmd.split("_")[-1])
            self.rotate_by_degrees(-deg)  # 우회전은 -
            return
        if cmd.startswith("TURN_LEFT_"):
            deg = float(cmd.split("_")[-1])
            self.rotate_by_degrees(+deg)
            return

        # 1) 카드 검색 계열 액션 처리
        SEARCH_MAP = {
            "SEARCH_APPLE":   "Apple",
            "SEARCH_ORANGE":   "Orange",   
            "SEARCH_Banana": "Banana",
            "SEARCH_WATERMELON":    "Watermelon",
        }

        if cmd in SEARCH_MAP:
            target = SEARCH_MAP[cmd]
            self.search_target_label = target

            # 제자리 회전 시작 (좌우 반대 방향으로)
            self.data.ctrl[0] = 4.0
            self.data.ctrl[1] = -4.0

            self.current_action = cmd
            # 검색 모드는 duration으로 멈추지 않게, action_end_sim_time은 무시
            self.action_end_sim_time = float("inf")

            print(f"[TurtlebotFactorySim] Start search for '{target}' (cmd={cmd})")
            return

        # 2) 일반 ACTION_TABLE 기반 액션 처리
        if cmd not in ACTION_TABLE:
            print(f"[TurtlebotFactorySim] Unknown command: {cmd}")
            return

        duration = base_duration
        if cmd in ["좌회전", "우회전"]:
            duration *= 1.6
        elif cmd == "제자리 회전":
            duration *= 1.0

        left, right = ACTION_TABLE[cmd]
        self.data.ctrl[0] = left
        self.data.ctrl[1] = right

        self.current_action = cmd
        self.action_end_sim_time = self.data.time + duration

        print(f"[TurtlebotFactorySim] Command '{cmd}' → L={left}, R={right}, duration={duration:.2f}s")

    def _process_commands(self):
        """command_queue 에 쌓인 명령들을 한 번에 처리."""
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            self.apply_command(cmd)

    def yolo_detect_dict(self):
        if (not self.use_yolo) or (self.detector is None) or (self.latest_frame is None):
            return {}
        return self.detector.detect_dict(self.latest_frame)

    def yolo_detect_image(self):
        if (not self.use_yolo) or (self.detector is None) or (self.latest_frame is None):
            return None
        return self.detector.detect_image(self.latest_frame)

    def get_front_distance(self) -> float:  #앞 물체와의 거리를 측정
        return float(self.data.sensordata[self.rf_front_adr])
    
    def get_wheel_encoder(self):    #wheel encoder의 정보를 가져옴
        """
        Returns:
            l_pos, r_pos: wheel angle [rad]
            l_vel, r_vel: wheel angular velocity [rad/s]
        """
        l_pos = float(self.data.sensordata[self.enc_l_pos_adr])
        r_pos = float(self.data.sensordata[self.enc_r_pos_adr])
        l_vel = float(self.data.sensordata[self.enc_l_vel_adr])
        r_vel = float(self.data.sensordata[self.enc_r_vel_adr])
        return l_pos, r_pos, l_vel, r_vel
        
    def _run_yolo_on_latest_frame(self):
        if not self.use_yolo or self.detector is None:
            return
        img_bgr = self.yolo_detect_image()
        if img_bgr is None:
            return
        cv2.imshow(self.yolo_window_name, img_bgr)

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        print("[TurtlebotFactorySim] Start simulation loop.")
        try:
            while self._running and not self.viewer.should_close():
                # 1) 명령 처리
                self._process_commands()

                # 2) 시뮬레이션 한 스텝
                self.step_simulation()

                # 3) 렌더 + latest_frame 갱신
                self.render()

                # 계속 직진하라는 명령을 내렸을 경우 직진 action을 계속 실행
                if self.cruise_forward and (not self.auto_scan_mode):
                    # 계속 직진 컨트롤 유지(혹시 다른 액션이 ctrl을 건드렸을 때 대비)
                    self.data.ctrl[0] = self.cruise_speed
                    self.data.ctrl[1] = self.cruise_speed

                # 직진 도중 장애물 감지 -> 1m 미만이면 직진 중지 + 자동 탐색 시작
                if self.cruise_forward and (not self.auto_scan_mode):
                    front = float(self.get_front_distance())
                    if front < self.avoid_threshold_m:
                        print(f"[AUTO_AVOID] front={front:.2f}m < {self.avoid_threshold_m:.2f}m → stop & scan")
                        # 직진 정지
                        self.data.ctrl[0] = 0.0
                        self.data.ctrl[1] = 0.0
                        self.cruise_forward = False

                        # 스캔 시작
                        self.auto_scan_mode = True
                        self.scan_stage = 0
                        self.scan_d_right = None
                        self.scan_d_left = None
                        self.current_action = None
                        self.action_end_sim_time = 0.0

                # 탐색 모드
                if self.auto_scan_mode:
                    # (rotate_by_degrees는 블로킹이라 start 루프를 잠깐 잡아먹지만, 스캔에는 오히려 단순해서 OK)
                    if self.scan_stage == 0:    #오른쪽으로 60도 회전
                        self.rotate_by_degrees_fast(-60)
                        self.scan_stage = 1

                    elif self.scan_stage == 1:  #전방 물체와 거리 측정
                        self.scan_d_right = float(self.get_front_distance())
                        print(f"[AUTO_AVOID] d_right={self.scan_d_right:.3f} m")
                        self.scan_stage = 2

                    elif self.scan_stage == 2:  #왼쪽으로 120도 회전 -> 처음 대비 왼쪽 60도
                        self.rotate_by_degrees_fast(+120)
                        self.scan_stage = 3

                    elif self.scan_stage == 3:
                        self.scan_d_left = float(self.get_front_distance()) #전방 물체와 거리 측정
                        print(f"[AUTO_AVOID] d_left={self.scan_d_left:.3f} m")
                        self.scan_stage = 4

                    elif self.scan_stage == 4:  #다시 60도 오른쪽으로 -> 처음 각도로 복귀
                        dr, dl = self.scan_d_right, self.scan_d_left
                        print(f"[AUTO_AVOID] compare: right={dr:.3f} vs left={dl:.3f}")

                        print("[AUTO_AVOID] TURN_RIGHT_60")
                        self.rotate_by_degrees_fast(-60)
                        self.scan_stage = 5

                    elif self.scan_stage == 5:  #측정 거리를 기반으로 어떤 행동을 할지 결정
                        dr, dl = self.scan_d_right, self.scan_d_left

                        if dr > dl:
                            print("[AUTO_AVOID] choose RIGHT -> TURN_RIGHT_90")
                            self.rotate_by_degrees(-90)
                        elif dl > dr:
                            print("[AUTO_AVOID] choose LEFT -> TURN_LEFT_90")
                            self.rotate_by_degrees(+90)
                        else:
                            if (dr < 1.2) and (dl < 1.2):
                                print("[AUTO_AVOID] tie & both < 1.2m -> TURN_LEFT_180")
                                self.rotate_by_degrees(+180)
                            else:
                                print("[AUTO_AVOID] tie & at least one >= 1.2m -> TURN_LEFT_90")
                                self.rotate_by_degrees(+90)

                        self.auto_scan_mode = False
                        print("[AUTO_AVOID] done")
                        self.cruise_forward = True  #다시 계속 직진으로 전환
                        print("[AUTO_AVOID] resume cruise")

                # 3.5) 검색 모드라면: YOLO로 타겟 감시
                if self.search_target_label is not None:
                    det = self.yolo_detect_dict()
                    if self.search_target_label in det:
                        # 타겟 발견 → 정지 + 검색 종료
                        self.data.ctrl[0] = 0.0
                        self.data.ctrl[1] = 0.0
                        print(f"[TurtlebotFactorySim] Found '{self.search_target_label}' → stop search.")
                        self.search_target_label = None
                        self.current_action = None
                        self.action_end_sim_time = 0.0

                # 4) 일반 액션 duration 기반 정지 (검색 모드일 땐 X)
                if (
                    self.current_action 
                    and not (self.current_action.startswith("SEARCH_"))
                    and self.data.time > self.action_end_sim_time
                ):
                    self.data.ctrl[0] = 0.0
                    self.data.ctrl[1] = 0.0
                    print(f"[TurtlebotFactorySim] '{self.current_action}' 완료 → stop.")
                    self.current_action = None

                # 5) YOLO 디스플레이
                if self.use_yolo:
                    self._run_yolo_on_latest_frame()

                # 6) q로 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[TurtlebotFactorySim] 'q' 입력으로 종료합니다.")
                    break

        except Exception as e:
            print(f"\n[TurtlebotFactorySim] 시뮬레이션 중 예외 발생: {e}")
        finally:
            self.close()

    def close(self):
        """시뮬레이션 종료 및 리소스 정리."""
        self._running = False
        if self.use_yolo:
            cv2.destroyWindow(self.yolo_window_name)
        self.viewer.terminate()
        print("[TurtlebotFactorySim] Simulation terminated.")
