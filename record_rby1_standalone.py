#!/usr/bin/env python

"""
RBY1 SDK LeRobot 형식 데이터 로깅

현재 로봇 상태(조인트 + 그리퍼 + 카메라)를 LeRobot 데이터셋 형식으로 기록합니다.

=== 두 가지 모드 ===

1. 관측 전용 모드 (기본, --teleop 없음):
   - 로봇 제어권 없이 상태만 읽음
   - SDK teleoperation과 동시 실행 가능
   - 터미널 1: SDK teleoperation 실행 (로봇 제어)
   - 터미널 2: 이 스크립트 실행 (녹화만)

2. 텔레오퍼레이션 모드 (--teleop):
   - 마스터 암으로 로봇을 직접 제어하며 녹화
   - 제어권 획득, 그리퍼/마스터 암 초기화
   - SDK teleoperation 없이 단독 실행

키보드 조작:
    SPACE : 녹화 시작/중지 토글
    ENTER : 현재 에피소드 저장하고 다음 에피소드로
    R     : 현재 에피소드 취소하고 다시 녹화
    B     : 이전 에피소드 삭제하고 재녹화
    Q     : 종료

    헤드 제어 (teleop 모드에서만):
    W/S   : 헤드 위/아래 (tilt)
    A/D   : 헤드 좌/우 (pan)
    X     : 헤드 중앙으로 리셋

사용 방법:
    # 관측 전용 모드 (SDK teleoperation과 함께 사용)
    # 터미널 1: python rby1-sdk/examples/python/99_teleoperation_with_joint_mapping.py --address 192.168.30.1:50051
    # 터미널 2:
    python record_rby1_standalone.py --address 192.168.30.1:50051 --episodes 10

    # 텔레오퍼레이션 모드 (단독 실행, 마스터 암으로 로봇 제어)
    python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --episodes 5

    # 카메라 포함
    python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --episodes 5
"""

import argparse
import logging
import os
import time
import signal
import sys
import threading
import termios
import tty
import select
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import numpy as np

# 로깅 설정 (17_teleop과 동일)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# WebUI 템플릿 import
from webui_template import generate_html, generate_camera_div

try:
    import rby1_sdk as rby
    import rby1_sdk.dynamics as rby_dyn
except ImportError:
    print("ERROR: rby1_sdk를 찾을 수 없습니다.")
    print("rby1-sdk를 먼저 빌드/설치하세요.")
    sys.exit(1)

# LeRobot 데이터셋 사용
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================================================================
# 설정
# ============================================================================

# 에피소드당 최대 시간 (초) - 5분
MAX_EPISODE_DURATION = 300

# RBY1-A 조인트 이름 (팔별로 분리)
RIGHT_ARM_JOINTS = [
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
    "right_arm_4", "right_arm_5", "right_arm_6",
]

LEFT_ARM_JOINTS = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
    "left_arm_4", "left_arm_5", "left_arm_6",
]

# [개발중] 휠 조인트 이름
WHEEL_JOINTS = [
    "wheel_0",  # 왼쪽 휠
    "wheel_1",  # 오른쪽 휠
]

# ============================================================================
# RealSense 카메라 시리얼 번호 ↔ 이름 매핑
# 시리얼 번호는 재부팅해도 변하지 않으므로 안정적인 매핑 가능
# ============================================================================
CAMERA_SERIAL_MAP = {
    "315122272205": "cam_left_wrist",   # D405 - 왼손
    "335122271196": "cam_right_wrist",  # D405 - 오른손
    # D435i 헤드 카메라는 시리얼 번호 확인 후 추가
    # "XXXXXXXXXX": "cam_high",  # D435i - 헤드
}

# 카메라 모델명으로 자동 감지 (시리얼 매핑이 없을 때 fallback)
CAMERA_MODEL_MAP = {
    "D435i": "cam_high",      # D435i는 헤드 카메라로 자동 할당
    "D435": "cam_high",       # D435도 헤드로
}


# ============================================================================
# 텔레오퍼레이션 설정 (SDK에서 가져옴)
# 참조: rby1-sdk/examples/python/17_teleoperation_with_joint_mapping.py
# ============================================================================

class TeleopSettings:
    """텔레오퍼레이션 설정
    
    17_teleop 기본값과 동일:
        master_arm_loop_period = 1 / 100
        impedance_stiffness = 50
        impedance_damping_ratio = 1.0
        impedance_torque_limit = 30.0
    """
    master_arm_loop_period = 1 / 100  # 100Hz (17_teleop 기본값: 1/100)
    impedance_stiffness = 50          # (17_teleop 기본값: 50)
    impedance_damping_ratio = 1.0     # (17_teleop 기본값: 1.0)
    impedance_torque_limit = 30.0     # (17_teleop 기본값: 30.0)
    
    # ========================================================================
    # 마스터 암 안전 모니터링 임계값
    # ========================================================================
    # 마스터 암 모터 배치 (7개 관절 × 2팔):
    #   관절 0-2: XM540-W150 (Stall 7.3Nm, 권장 연속 사용 ~3.5Nm)
    #   관절 3-6: XM430-W210 (Stall 3.0Nm, 권장 연속 사용 ~1.5Nm)
    # state.torque_joint = current × torque_constant (SDK에서 계산됨)
    
    # 마스터 암 토크 임계값 (Nm) - 관절별
    ma_torque_warning = np.array([
        2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 1.0,  # 오른팔 (70% of limit)
        2.5, 2.5, 2.5, 2.5, 1.0, 1.0, 1.0,  # 왼팔
    ])
    ma_torque_critical = np.array([
        3.5, 3.5, 3.5, 3.5, 1.5, 1.5, 1.5,  # 오른팔 (MA_TORQUE_LIMIT과 동일)
        3.5, 3.5, 3.5, 3.5, 1.5, 1.5, 1.5,  # 왼팔
    ])
    
    # 하위 호환성용 (WebUI에서 사용할 수 있음)
    temp_warning = 60
    temp_critical = 70
    current_warning = 10.0
    current_critical = 20.0
    torque_warning = 30.0
    torque_critical = 50.0

# ============================================================================
# 초기 자세 설정
# ============================================================================
# 17_teleop 기본값 (Ready pose):
#   torso:     [0.0, 45.0, -90.0, 45.0, 0.0, 0.0] deg
#   right_arm: [0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0] deg
#   left_arm:  [0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0] deg
#
# 아래는 커스텀 자세 예시들:
#   Packing:   torso [0.0, 80.0, -140.0, 60.0, 0.0, 0.0] deg  ← 현재 "A" 모델에 적용됨
#   중간허리:  torso [0.0, 55.0, -110.0, 50.0, 0.0, 0.0] deg
#   전투모드:  torso [0.0, 67.8, -82.8, 31.8, 0.0, 0.0] deg (라디안으로 저장된 값 변환)
# ============================================================================

# 초기 자세 (모델별)
READY_POSE = {
    "A": {
        # ⚠️ 변경됨: 17_teleop 기본값 [0,45,-90,45,0,0] → Packing 자세 [0,80,-140,60,0,0]
        "torso": np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]),
        "right_arm": np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),  # 17_teleop 기본값과 동일
        "left_arm": np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),   # 17_teleop 기본값과 동일
        # 전투모드 (주석):
        #"torso": np.array([0.0,1.1839635825151906,-1.4456515921713253,0.5552402935002304,0.0,0.0,]),
        #"right_arm": np.array([-0.015897964254646485,-1.6672738461993182,-0.3115309943159733,-1.1695426443162062,0.7229574754265632,-1.3463979472390455,0.0,]),
        #"left_arm": np.array([0.00019364608955982105,1.679986142431598,0.3165619956623804,-1.1723713960166389,-0.7150267947531944,-1.271152354641285,0.0,]),
    },
    "M": {
        "torso": np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]),           # 17_teleop 기본값과 동일
        "right_arm": np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]), # 17_teleop 기본값과 동일
        "left_arm": np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),   # 17_teleop 기본값과 동일
    },
}

# ============================================================================
# 마스터 암 관절 제한 (17_teleop 기본값과 동일)
# ============================================================================
MA_Q_LIMIT_BARRIER = 0.5  # (17_teleop 기본값: 0.5)
# 마스터 암 관절 각도 제한 [오른팔 7 + 왼팔 7]
MA_MIN_Q = np.deg2rad([-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360])  # 17_teleop 기본값
MA_MAX_Q = np.deg2rad([360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360])           # 17_teleop 기본값
# 마스터 암 토크 제한 (XM540: 3.5Nm, XM430: 1.5Nm)
MA_TORQUE_LIMIT = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)  # 17_teleop 기본값
# 마스터 암 점성 게인 (관절별 damping)
MA_VISCOUS_GAIN = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)  # 17_teleop 기본값

# 그리퍼 방향 설정 (17_teleop 기본값: False = 반전)
# True: 정방향 (trigger 증가 → 그리퍼 닫힘)
# False: 반전 (trigger 증가 → 그리퍼 열림)
GRIPPER_DIRECTION = False


class Gripper:
    """그리퍼 제어 클래스 (SDK에서 가져옴)"""
    
    def __init__(self):
        self.bus = None
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q = None
        self._running = False
        self._thread = None
    
    def initialize(self):
        """그리퍼 초기화"""
        try:
            self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
            self.bus.open_port()
            self.bus.set_baud_rate(2_000_000)
            self.bus.set_torque_constant([1, 1])
            
            rv = True
            for dev_id in [0, 1]:
                if not self.bus.ping(dev_id):
                    print(f"⚠ Dynamixel ID {dev_id} 응답 없음")
                    rv = False
            
            if rv:
                self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
                print("✓ 그리퍼 초기화 완료")
            return rv
        except Exception as e:
            print(f"⚠ 그리퍼 초기화 실패: {e}")
            return False
    
    def set_operating_mode(self, mode):
        """그리퍼 작동 모드 설정"""
        if self.bus is None:
            return
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
    
    def homing(self):
        """그리퍼 홈 위치 탐색"""
        if self.bus is None:
            return
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 0.5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            # 99_teleoperation과 동일하게 group_fast_sync_read_encoder 사용
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q.copy()
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        
        self.target_q = self.max_q.copy()
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        print(f"✓ 그리퍼 홈 완료 (범위: {self.min_q} ~ {self.max_q})")
    
    def set_target(self, target: np.ndarray):
        """그리퍼 목표 위치 설정 (0-1 범위)
        
        GRIPPER_DIRECTION에 따라 방향 결정:
            True: 정방향 (trigger 증가 → 그리퍼 닫힘)
            False: 반전 (trigger 증가 → 그리퍼 열림)
        """
        # min_q/max_q 유효성 체크 (17_teleop과 동일)
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            logging.error("Cannot set target. min_q or max_q is not valid.")
            return
        
        normalized_q = np.clip(target, 0, 1)
        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q
    
    def start(self):
        """그리퍼 제어 스레드 시작"""
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """그리퍼 제어 스레드 정지"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _control_loop(self):
        """그리퍼 제어 루프 (99_teleoperation과 동일)"""
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            if self.bus and self.target_q is not None:
                try:
                    self.bus.group_sync_write_send_position(
                        [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                    )
                except Exception:
                    pass
            time.sleep(0.1)  # 10Hz (99_teleoperation과 동일)


class KeyboardController:
    """비차단 키보드 입력 처리"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout: float = 0.01) -> Optional[str]:
        """비차단으로 키 입력 확인"""
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None


class RBY1Recorder:
    """RBY1 SDK를 사용한 LeRobot 형식 데이터 레코더"""

    def __init__(self, address: str, model: str = "a", camera_id: Optional[int] = None, 
                 arms: str = "both", use_realsense: bool = False, use_teleop: bool = False,
                 camera_names: Optional[list] = None, stream_port: int = 0,
                 control_mode: str = "impedance", reset_pose: bool = True,
                 use_wheels: bool = False):
        self.address = address
        self.model = model
        self.camera_id = camera_id
        self.arms = arms
        self.use_realsense = use_realsense
        self.use_teleop = use_teleop
        self.stream_port = stream_port  # 웹 스트리밍 포트 (0이면 비활성화)
        self.control_mode = control_mode  # 'position' 또는 'impedance'
        self.position_mode = (control_mode == "position")
        self.reset_pose_each_episode = reset_pose  # 에피소드마다 초기 자세로 리셋
        self.use_wheels = use_wheels  # [개발중] 휠 데이터 기록 여부
        
        # 카메라 이름 설정: arms에 따라 기본값 결정
        if camera_names is not None:
            self.camera_names = camera_names
        else:
            self.camera_names = self._get_default_camera_names(arms)

        self.robot = None
        self.camera = None
        
        # 웹 스트리밍 관련
        self.stream_server = None
        self.stream_frames = {}  # {camera_name: frame}
        self.stream_lock = threading.Lock()
        
        # 멀티 RealSense 카메라 지원
        self.rs_pipelines = {}  # {camera_name: (pipeline, serial)}
        self.rs_pipeline = None  # 하위 호환성 유지
        
        # 마스터 암 관련
        self.master_arm = None
        self.master_arm_state = None
        self.master_arm_lock = threading.Lock()
        
        # 텔레오퍼레이션 관련
        self.command_stream = None
        self.gripper = None
        self.right_q = None  # 오른팔 목표 위치
        self.left_q = None   # 왼팔 목표 위치
        self.robot_q = None  # 현재 로봇 관절 위치
        
        # 로봇 관절 제한 (텔레오퍼레이션에서 초기화됨)
        self.robot_max_q = None
        self.robot_min_q = None
        self.robot_max_qdot = None
        self.robot_max_qddot = None
        self.right_minimum_time = 1.0
        self.left_minimum_time = 1.0
        
        # 헤드 제어 관련
        self.head_q = np.array([0.0, 0.0])  # [pan (head_0), tilt (head_1)]
        self.head_limits = {
            'pan': (-0.523, 0.523),    # head_0: -30° ~ 30°
            'tilt': (-0.35, 1.57),     # head_1: -20° ~ 90°
        }
        self.head_step = np.deg2rad(5.0)  # 키 한번에 5도 이동

        # 상태 데이터
        self.latest_state = None
        self.state_lock = threading.Lock()
        self.running = False
        
        # 시그널 핸들러 관련
        self._shutdown_requested = False
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        
        # 셧다운 중복 호출 방지 플래그
        self._master_arm_stopped = False
        self._robot_control_cancelled = False
        self._state_update_stopped = False
        
        # 안전 모니터링: 마스터 암 토크 기반
        # state.torque_joint = current × torque_constant (SDK에서 계산됨)
        self._teleop_paused = False  # Critical 감지 시 teleop 일시정지
        self._critical_reason = ""   # 일시정지 사유
        self._ma_warning_count = 0   # 경고 로그 빈도 제한용
        self._ma_disconnect_requested = False  # 토크 과부하 시 마스터암 해제 요청
        
        # 로그 폴더/파일 설정 (시간 기반 폴더명)
        self._log_dir = None
        self._log_file = None
        self._ma_log_count = 0

        # 선택한 팔에 따른 조인트 이름 설정
        self.joint_names = self._get_joint_names(arms)

    def _get_default_camera_names(self, arms: str) -> list:
        """팔 선택에 따른 기본 카메라 이름 반환"""
        if arms == "right":
            # head + right wrist (2대)
            return ["cam_high", "cam_right_wrist"]
        elif arms == "left":
            # head + left wrist (2대)
            return ["cam_high", "cam_left_wrist"]
        else:  # both
            # head + left wrist + right wrist (3대)
            return ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    @property
    def has_camera(self) -> bool:
        """카메라 연결 여부 (RealSense 또는 일반 카메라)"""
        return len(self.rs_pipelines) > 0 or self.camera is not None
    
    @property
    def num_cameras(self) -> int:
        """연결된 카메라 수"""
        if self.rs_pipelines:
            return len(self.rs_pipelines)
        elif self.camera is not None:
            return 1
        return 0
    
    @property
    def active_camera_names(self) -> list[str]:
        """활성화된 카메라 이름 목록"""
        if self.rs_pipelines:
            return list(self.rs_pipelines.keys())
        elif self.camera is not None:
            return ["camera"]
        return []

    def _get_joint_names(self, arms: str) -> list[str]:
        """선택한 팔에 따른 조인트 이름 반환"""
        if arms == "right":
            joints = RIGHT_ARM_JOINTS.copy()
        elif arms == "left":
            joints = LEFT_ARM_JOINTS.copy()
        elif arms == "both":
            joints = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
        else:
            raise ValueError(f"Invalid arms option: {arms}. Use 'right', 'left', or 'both'")
        
        # [개발중] 휠 조인트 추가
        if self.use_wheels:
            joints = joints + WHEEL_JOINTS
        
        return joints

    def _state_callback(self, robot_state, control_manager_state=None):
        """로봇 상태 업데이트 콜백"""
        with self.state_lock:
            self.latest_state = robot_state
            # 텔레오퍼레이션용 로봇 관절 위치 업데이트
            if robot_state is not None:
                self.robot_q = np.array(robot_state.position)

    def _setup_log_folder(self):
        """로그 폴더 및 파일 설정 (시간 기반 폴더명)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_base = Path.home() / "vla_ws" / "logs"
        self._log_dir = log_base / f"teleop_{timestamp}"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # 텔레옵 로그 파일 (마스터암 상태 + 토크)
        log_path = self._log_dir / "teleop_state.log"
        self._log_file = open(log_path, "w")
        self._log_file.write(f"# Teleop State Log - {timestamp}\n")
        self._log_file.write("# Format: timestamp,btn_R,btn_L,trig_R,trig_L,torque_R[0-6],torque_L[0-6]\n")
        self._log_file.write("# Torque unit: Nm, XM540(0-2) limit 3.5Nm, XM430(3-6) limit 1.5Nm\n")
        self._log_file.flush()
        
        # 안전 이벤트 로그 파일 (경고/위험 기록)
        safety_log_path = self._log_dir / "safety_events.log"
        self._safety_log_file = open(safety_log_path, "w")
        self._safety_log_file.write(f"# Safety Events Log - {timestamp}\n")
        self._safety_log_file.write("# Format: timestamp,level,message\n")
        self._safety_log_file.flush()
        
        print(f"📁 로그 폴더: {self._log_dir}")
    
    def _write_teleop_log(self, state):
        """텔레옵 상태를 로그 파일에 기록 (버튼 + 토크)"""
        if self._log_file is None:
            return
        
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            # 버튼/트리거 상태
            btn_data = f"{state.button_right.button},{state.button_left.button},{state.button_right.trigger:.2f},{state.button_left.trigger:.2f}"
            # 토크 데이터 (7개씩 오른팔/왼팔)
            torque = state.torque_joint
            torque_r = ",".join(f"{torque[i]:.3f}" for i in range(7))    # Right arm: 0-6
            torque_l = ",".join(f"{torque[i]:.3f}" for i in range(7, 14))  # Left arm: 7-13
            line = f"{timestamp},{btn_data},{torque_r},{torque_l}\n"
            self._log_file.write(line)
            self._log_file.flush()
        except Exception:
            pass  # 로깅 실패는 무시
    
    def _write_safety_log(self, level: str, message: str):
        """안전 이벤트를 로그 파일에 기록
        
        Args:
            level: 'WARNING' 또는 'CRITICAL'
            message: 이벤트 메시지
        """
        if not hasattr(self, '_safety_log_file') or self._safety_log_file is None:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            line = f"{timestamp},{level},{message}\n"
            self._safety_log_file.write(line)
            self._safety_log_file.flush()
        except Exception:
            pass  # 로깅 실패는 무시
    
    def _close_log_file(self):
        """로그 파일 닫기"""
        if self._log_file is not None:
            try:
                self._log_file.close()
                self._log_file = None
            except Exception:
                pass
        
        # 안전 이벤트 로그 파일도 닫기
        if hasattr(self, '_safety_log_file') and self._safety_log_file is not None:
            try:
                self._safety_log_file.close()
                self._safety_log_file = None
            except Exception:
                pass

    def _read_master_arm_motor_states(self, label: str = "snapshot"):
        """마스터암 다이나믹셀 온도/전류 스냅샷
        
        MasterArm.State에는 torque_joint만 있고 temperature/current는 없음.
        시작 전/종료 후에 DynamixelBus를 직접 열어 스냅샷 기록.
        
        Args:
            label: 로그에 기록할 레이블 ("start", "end" 등)
        """
        bus = None
        try:
            bus = rby.DynamixelBus(rby.upc.MasterArmDeviceName)
            motor_ids = list(range(14))  # 0-13: 14개 관절
            
            states = bus.get_motor_states(motor_ids)
            if states is None:
                self._write_safety_log("INFO", f"{label}: 마스터암 모터 상태 읽기 실패")
                return None
            
            # 결과 파싱
            motor_data = {}
            for motor_id, ms in states:
                motor_data[motor_id] = {
                    'temperature': ms.temperature,
                    'current': ms.current,
                    'torque': ms.torque,
                }
            
            # 로그 기록
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp_str = ",".join(f"{motor_data.get(i, {}).get('temperature', 0)}" for i in range(14))
            curr_str = ",".join(f"{motor_data.get(i, {}).get('current', 0):.3f}" for i in range(14))
            
            self._write_safety_log("INFO", f"{label}: temp[0-13]={temp_str}")
            self._write_safety_log("INFO", f"{label}: curr[0-13]={curr_str}")
            
            # 경고: 온도 50°C 이상
            for i in range(14):
                temp = motor_data.get(i, {}).get('temperature', 0)
                if temp >= 50:
                    arm = "Right" if i < 7 else "Left"
                    joint = i if i < 7 else i - 7
                    self._write_safety_log("WARNING", f"{label}: {arm} joint {joint} temperature {temp}°C >= 50°C")
            
            print(f"🌡️  마스터암 온도 ({label}): {temp_str}")
            return motor_data
            
        except Exception as e:
            self._write_safety_log("INFO", f"{label}: 마스터암 상태 읽기 오류 - {e}")
            return None
        finally:
            # 버스 명시적 해제 (GC 대기 없이 즉시 해제)
            if bus is not None:
                del bus
                time.sleep(0.1)  # 버스 해제 대기

    def _disconnect_master_arm_safe(self):
        """토크 과부하로 인한 마스터암 안전 해제 (별도 스레드에서 호출)
        
        콜백 내부에서 직접 stop_control() 호출 시 데드락 발생하므로
        별도 스레드에서 안전하게 해제
        """
        if not self._ma_disconnect_requested:
            return
            
        try:
            # 잠시 대기 (콜백 완료 대기)
            time.sleep(0.1)
            
            if self.master_arm is not None and not self._master_arm_stopped:
                self.master_arm.stop_control()
                self._master_arm_stopped = True
                print("\n" + "=" * 60)
                print("🛑 마스터암 토크 과부하로 연결 해제됨!")
                print(f"   사유: {self._critical_reason}")
                print("=" * 60)
                self._write_safety_log("CRITICAL", "마스터암 연결 해제 완료")
                
                # 종료 시점 온도/전류 스냅샷
                self._read_master_arm_motor_states("EMERGENCY_END")
                
        except Exception as e:
            self._write_safety_log("CRITICAL", f"마스터암 해제 오류: {e}")

    def connect(self):
        """로봇 및 카메라, 마스터 암 연결"""
        print(f"로봇 연결 중: {self.address}")
        self.robot = rby.create_robot(self.address, self.model)
        self.robot.connect()

        if not self.robot.is_connected():
            raise ConnectionError("로봇 연결 실패") 

        print("✓ 로봇 연결됨")

        # 텔레오퍼레이션 모드: 제어권 획득
        if self.use_teleop:
            # 파워 상태 확인 (필요시 파워온)
            if not self.robot.is_power_on(".*"):
                print("파워 온 중...")
                if not self.robot.power_on(".*"):
                    raise RuntimeError("파워 온 실패")
                print("✓ 파워 온 완료")
            
            # 서보 온 (팔 + 헤드)
            servo_pattern = "torso_.*|right_arm_.*|left_arm_.*|head_.*"
            if not self.robot.is_servo_on(servo_pattern):
                print("서보 온 중...")
                if not self.robot.servo_on(servo_pattern):
                    raise RuntimeError("서보 온 실패")
                print("✓ 서보 온 완료 (팔 + 헤드)")
            
            # Control Manager 활성화
            self.robot.reset_fault_control_manager()
            if not self.robot.enable_control_manager():
                raise RuntimeError("Control Manager 활성화 실패")
            print("✓ Control Manager 활성화")
            
            # 12V 출력 (그리퍼용)
            for arm in ["right", "left"]:
                if not self.robot.set_tool_flange_output_voltage(arm, 12):
                    print(f"⚠ Tool flange 전압 설정 실패 ({arm})")
            
            # 저역통과 필터 설정 (17_teleop 기본값: 3)
            self.robot.set_parameter("joint_position_command.cutoff_frequency", "3")
            
            # Command stream 생성
            self.command_stream = self.robot.create_command_stream(priority=1)
            print("✓ 텔레오퍼레이션 제어권 획득")
        else:
            # 관측 전용 모드: 제어권 없이 상태만 읽음
            print("📡 관측 전용 모드 (제어권 없음)")
            print("   → SDK teleoperation과 동시 실행 가능")
            if not self.robot.is_power_on(".*"):
                print("⚠ 로봇 파워가 꺼져 있습니다. SDK teleoperation을 먼저 실행하세요.")
            self.command_stream = None

        # 상태 스트리밍 시작
        self.robot.start_state_update(self._state_callback, rate=100)

        # 첫 상태 수신 대기
        timeout = 5.0
        start = time.time()
        while self.latest_state is None:
            if time.time() - start > timeout:
                raise TimeoutError("로봇 상태 수신 타임아웃")
            time.sleep(0.01)
        print("✓ 상태 스트리밍 시작됨")

        # Dynamics 모델 초기화 (EEF pose 계산용)
        try:
            self.robot_model = self.robot.model()
            self.dyn_robot = self.robot.get_dynamics()
            
            # EEF 링크 이름 설정
            eef_links = ["base"]
            if self.arms in ["right", "both"]:
                eef_links.append("ee_right")
            if self.arms in ["left", "both"]:
                eef_links.append("ee_left")
            
            self.dyn_state = self.dyn_robot.make_state(eef_links, self.robot_model.robot_joint_names)
            print(f"✓ Dynamics 모델 초기화 (EEF: {eef_links[1:]})")
        except Exception as e:
            print(f"⚠ Dynamics 모델 초기화 실패: {e}")
            print("  EEF pose 기록이 비활성화됩니다.")
            self.dyn_robot = None

        # 카메라 연결
        if self.camera_id is not None or self.use_realsense:
            self._connect_camera()

        # 웹 스트리밍 서버 시작
        if self.stream_port > 0:
            self._start_stream_server()

        # 마스터 암 및 그리퍼 연결 (teleop 모드)
        if self.use_teleop:
            self._setup_log_folder()  # 로그 폴더 설정
            self._setup_teleop()
        
        # 시그널 핸들러 등록 (안전한 종료를 위해)
        self._register_signal_handlers()
        print("✓ 시그널 핸들러 등록 완료 (Ctrl+C로 안전 종료)")

    def _register_signal_handlers(self):
        """시그널 핸들러 등록 (SIGINT, SIGTERM)"""
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _restore_signal_handlers(self):
        """원래 시그널 핸들러 복원"""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
    
    def _signal_handler(self, signum, frame):
        """긴급 정지 시그널 핸들러 (Ctrl+C, SIGTERM)"""
        if self._shutdown_requested:
            # 두 번째 시그널: 강제 종료
            print("\n⛔ 강제 종료!")
            sys.exit(1)
        
        self._shutdown_requested = True
        print("\n")
        print("=" * 60)
        print("⚠️  긴급 정지 요청 (Ctrl+C)")
        print("=" * 60)
        print("안전하게 종료 중...")
        
        # 녹화 중지
        self.running = False
        
        # 1. 상태 업데이트 중지 (17_teleop과 동일: 가장 먼저)
        if self.robot is not None and not self._state_update_stopped:
            try:
                self.robot.stop_state_update()
                self._state_update_stopped = True
                print("  ✓ 상태 업데이트 중지")
            except Exception as e:
                print(f"  ⚠ 상태 업데이트 중지 실패: {e}")
        
        # 2. 마스터 암 중지
        if self.master_arm is not None and not self._master_arm_stopped:
            try:
                self.master_arm.stop_control()
                self._master_arm_stopped = True
                print("  ✓ 마스터 암 중지")
            except Exception as e:
                print(f"  ⚠ 마스터 암 중지 실패: {e}")
        
        # 3. 로봇 제어 취소
        if self.robot is not None and not self._robot_control_cancelled:
            try:
                self.robot.cancel_control()
                self._robot_control_cancelled = True
                print("  ✓ 로봇 제어 취소")
            except Exception as e:
                print(f"  ⚠ 로봇 제어 취소 실패: {e}")
        
        # 잠시 대기 후 정리
        time.sleep(0.5)
        
        # 전체 정리
        try:
            self.disconnect()
        except Exception as e:
            print(f"  ⚠ disconnect 중 오류: {e}")
        
        print("=" * 60)
        print("종료 완료")
        print("=" * 60)
        sys.exit(0)

    def _get_motor_status(self):
        """로봇 및 마스터 암 모터 상태 수집"""
        import json
        
        status = {
            "robot": {
                "connected": self.robot is not None and self.robot.is_connected(),
                "joints": [],
                "temperature": [],
                "current": [],
                "torque": [],
            },
            "master_arm": {
                "connected": self.master_arm is not None,
                "joints": [],
                "q_joint": [],
                "torque_joint": [],  # 마스터 암 토크 (모니터링용)
                "button_right": False,
                "button_left": False,
                "trigger_right": 0,
                "trigger_left": 0,
            },
            "gripper": {
                "connected": self.gripper is not None and self.gripper.bus is not None,
                "target_q": [],
                "min_q": [],
                "max_q": [],
            },
            "safety": {
                "teleop_paused": self._teleop_paused,
                "critical_reason": self._critical_reason,
            },
            "limits": {
                "temp_warning": TeleopSettings.temp_warning,
                "temp_critical": TeleopSettings.temp_critical,
                "current_warning": TeleopSettings.current_warning,
                "current_critical": TeleopSettings.current_critical,
                "torque_warning": TeleopSettings.torque_warning,
                "torque_critical": TeleopSettings.torque_critical,
                # 마스터 암 토크 임계값
                "ma_torque_warning": TeleopSettings.ma_torque_warning.tolist(),
                "ma_torque_critical": TeleopSettings.ma_torque_critical.tolist(),
            }
        }
        
        # 로봇 상태
        with self.state_lock:
            if self.latest_state is not None:
                state = self.latest_state
                # joint_states에서 온도, 전류, 토크 읽기
                if hasattr(state, 'temperature') and state.temperature is not None and len(state.temperature) > 0:
                    status["robot"]["temperature"] = [float(x) for x in state.temperature]
                if hasattr(state, 'current') and state.current is not None and len(state.current) > 0:
                    status["robot"]["current"] = [float(x) for x in state.current]
                if hasattr(state, 'torque') and state.torque is not None and len(state.torque) > 0:
                    status["robot"]["torque"] = [float(x) for x in state.torque]
                if hasattr(state, 'position') and state.position is not None and len(state.position) > 0:
                    status["robot"]["joints"] = [float(x) for x in state.position]
        
        # 마스터 암 상태
        with self.master_arm_lock:
            if self.master_arm_state is not None:
                ma_state = self.master_arm_state
                if hasattr(ma_state, 'q_joint'):
                    status["master_arm"]["q_joint"] = [float(x) for x in ma_state.q_joint]
                if hasattr(ma_state, 'torque_joint'):
                    status["master_arm"]["torque_joint"] = [float(x) for x in ma_state.torque_joint]
                if hasattr(ma_state, 'button_right'):
                    status["master_arm"]["button_right"] = bool(ma_state.button_right.button)
                    status["master_arm"]["trigger_right"] = int(ma_state.button_right.trigger)
                if hasattr(ma_state, 'button_left'):
                    status["master_arm"]["button_left"] = bool(ma_state.button_left.button)
                    status["master_arm"]["trigger_left"] = int(ma_state.button_left.trigger)
        
        # 그리퍼 상태
        if self.gripper is not None:
            if self.gripper.target_q is not None:
                status["gripper"]["target_q"] = [float(x) for x in self.gripper.target_q]
            if np.isfinite(self.gripper.min_q).all():
                status["gripper"]["min_q"] = [float(x) for x in self.gripper.min_q]
            if np.isfinite(self.gripper.max_q).all():
                status["gripper"]["max_q"] = [float(x) for x in self.gripper.max_q]
        
        return status

    def _start_stream_server(self):
        """웹 스트리밍 서버 시작 (MJPEG 멀티스레드 + 모터 모니터링)"""
        recorder = self
        from socketserver import ThreadingMixIn
        import cv2
        import json
        
        class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True
        
        class StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    # 연결된 카메라 목록으로 HTML 생성
                    cam_names = list(recorder.rs_pipelines.keys()) if recorder.rs_pipelines else []
                    if not cam_names:
                        with recorder.stream_lock:
                            cam_names = list(recorder.stream_frames.keys())
                    
                    camera_divs = "".join(generate_camera_div(name) for name in cam_names)
                    
                    # 임계값 설정
                    limits = {
                        "temp_warning": TeleopSettings.temp_warning,
                        "temp_critical": TeleopSettings.temp_critical,
                        "current_warning": TeleopSettings.current_warning,
                        "current_critical": TeleopSettings.current_critical,
                        "torque_warning": TeleopSettings.torque_warning,
                        "torque_critical": TeleopSettings.torque_critical,
                    }
                    
                    html = generate_html(camera_divs, recorder.stream_port, limits)
                    self.wfile.write(html.encode())
                
                elif self.path == '/api/status':
                    # JSON API 엔드포인트
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    status = recorder._get_motor_status()
                    self.wfile.write(json.dumps(status).encode())
                    
                elif self.path.endswith('.mjpeg'):
                    # MJPEG 스트리밍
                    cam_name = self.path[1:].replace('.mjpeg', '')
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    
                    try:
                        while True:
                            with recorder.stream_lock:
                                frame = recorder.stream_frames.get(cam_name)
                            
                            if frame is not None:
                                # RGB -> BGR for encoding
                                if len(frame.shape) == 3 and frame.shape[2] == 3:
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                else:
                                    frame_bgr = frame
                                _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                
                                self.wfile.write(b'--frame\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                                self.wfile.write(buffer.tobytes())
                                self.wfile.write(b'\r\n')
                            
                            time.sleep(0.033)  # ~30fps
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # 로그 숨기기
        
        def run_server():
            server = ThreadingHTTPServer(('0.0.0.0', recorder.stream_port), StreamHandler)
            recorder.stream_server = server
            server.serve_forever()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 스트리밍용 카메라 캡처 스레드 시작 (각 카메라별 별도 스레드)
        self._stream_running = True
        self._stream_capture_threads = []
        
        def stream_capture_loop(cam_name, pipeline):
            """각 카메라별 캡처 루프"""
            import pyrealsense2 as rs
            while self._stream_running:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=100)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame_rgb = np.asanyarray(color_frame.get_data())
                        with self.stream_lock:
                            self.stream_frames[cam_name] = frame_rgb.copy()
                except:
                    pass
                time.sleep(0.01)  # ~100fps max to reduce latency
        
        # 각 카메라에 대해 별도 스레드 시작
        for cam_name, (pipeline, _) in self.rs_pipelines.items():
            t = threading.Thread(target=stream_capture_loop, args=(cam_name, pipeline), daemon=True)
            t.start()
            self._stream_capture_threads.append(t)
        
        print(f"🌐 카메라 스트리밍: http://localhost:{self.stream_port}")

    def _connect_camera(self):
        """카메라 연결 (멀티 RealSense - 시리얼 번호 기반 동적 매핑)"""
        # RealSense 카메라 시도 (멀티 카메라 지원)
        if self.use_realsense:
            try:
                import pyrealsense2 as rs
                
                # 연결된 모든 RealSense 장치 검색
                ctx = rs.context()
                devices = ctx.query_devices()
                
                if len(devices) == 0:
                    print("⚠ RealSense 카메라를 찾을 수 없습니다.")
                else:
                    print(f"🔍 {len(devices)}개의 RealSense 카메라 감지됨")
                    
                    unassigned_idx = 0  # 매핑 안 된 카메라용 인덱스
                    
                    # 각 카메라에 파이프라인 생성
                    for device in devices:
                        serial = device.get_info(rs.camera_info.serial_number)
                        model_name = device.get_info(rs.camera_info.name)
                        
                        # 1. 시리얼 번호로 카메라 이름 결정 (최우선)
                        if serial in CAMERA_SERIAL_MAP:
                            cam_name = CAMERA_SERIAL_MAP[serial]
                        # 2. 모델명으로 자동 감지 (D435i → cam_high)
                        elif any(model in model_name for model in CAMERA_MODEL_MAP):
                            for model, default_name in CAMERA_MODEL_MAP.items():
                                if model in model_name:
                                    # 이미 할당된 이름인지 확인
                                    if default_name not in self.rs_pipelines:
                                        cam_name = default_name
                                    else:
                                        cam_name = f"{default_name}_{unassigned_idx}"
                                        unassigned_idx += 1
                                    break
                        # 3. 기본 이름 할당
                        else:
                            cam_name = f"camera_{unassigned_idx}"
                            unassigned_idx += 1
                        
                        try:
                            pipeline = rs.pipeline()
                            config = rs.config()
                            config.enable_device(serial)
                            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                            
                            pipeline.start(config)
                            self.rs_pipelines[cam_name] = (pipeline, serial)
                            print(f"  ✓ {cam_name}: {model_name} (S/N: {serial})")
                        except Exception as e:
                            print(f"  ⚠ {cam_name} 초기화 실패: {e}")
                    
                    # 하위 호환성: 첫 번째 파이프라인을 rs_pipeline에도 저장
                    if self.rs_pipelines:
                        first_name = list(self.rs_pipelines.keys())[0]
                        self.rs_pipeline = self.rs_pipelines[first_name][0]
                        print(f"✓ 총 {len(self.rs_pipelines)}개 RealSense 카메라 연결됨")
                        
                        # 매핑 요약 출력
                        print("📷 카메라 매핑:")
                        for cname, (_, cserial) in self.rs_pipelines.items():
                            print(f"   {cname} ← S/N: {cserial}")
                    
                    if self.rs_pipelines:
                        return
                        
            except ImportError:
                print("⚠ pyrealsense2가 설치되지 않음: pip install pyrealsense2")
                print("  일반 카메라로 시도합니다...")
            except Exception as e:
                print(f"⚠ RealSense 연결 실패: {e}")
                print("  일반 카메라로 시도합니다...")
        
        # 일반 USB 카메라 시도
        if self.camera_id is not None:
            try:
                import cv2
                self.camera = cv2.VideoCapture(self.camera_id)
                if not self.camera.isOpened():
                    print(f"⚠ 카메라 {self.camera_id} 열기 실패")
                    self.camera = None
                else:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"✓ 카메라 {self.camera_id} 연결됨")
            except ImportError:
                print("⚠ OpenCV 없음, 카메라 비활성화")
                self.camera = None

    def _setup_teleop(self):
        """완전한 텔레오퍼레이션 설정 (마스터 암 + 그리퍼 + 로봇 제어)"""
        try:
            model_name = self.robot_model.model_name if self.robot_model else "A"
            
            # 로봇 관절 제한 가져오기
            self.robot_max_q = self.dyn_robot.get_limit_q_upper(self.dyn_state)
            self.robot_min_q = self.dyn_robot.get_limit_q_lower(self.dyn_state)
            self.robot_max_qdot = self.dyn_robot.get_limit_qdot_upper(self.dyn_state)
            self.robot_max_qddot = self.dyn_robot.get_limit_qddot_upper(self.dyn_state)
            
            # Impedance 모드: 손목 관절 속도 제한 증가 (17_teleoperation과 동일)
            if not self.position_mode:
                self.robot_max_qdot[self.robot_model.right_arm_idx[-1]] *= 10
                self.robot_max_qdot[self.robot_model.left_arm_idx[-1]] *= 10
                print(f"✓ Impedance 모드 활성화 (stiffness={TeleopSettings.impedance_stiffness})")
            else:
                print("✓ Position 모드 활성화")
            
            # ========================================================================
            # ⚠️ 안전: 초기 자세로 이동 (17_teleop의 move_j처럼 블로킹, Position 모드)
            # 로봇이 이동 중에 마스터 암 제어가 시작되면 충돌 위험!
            # ========================================================================
            print("초기 자세로 이동 중...")
            ready_pose = READY_POSE.get(model_name, READY_POSE["A"])
            
            # 공식 17_teleop처럼 blocking으로 이동 (Position 모드)
            if not self._move_j(ready_pose, minimum_time=5.0):
                print("\n" + "=" * 60)
                print("⚠️  경고: 초기 자세 이동 실패!")
                print("   SDK에서 FinishCode.Ok를 반환하지 않았습니다.")
                print("=" * 60)
                # 3초 대기 (사용자가 상황 인지하도록)
                for i in range(3, 0, -1):
                    print(f"   {i}초 후 현재 위치에서 시작...", end="\r")
                    time.sleep(1)
                print("   현재 위치에서 시작합니다.          ")
            print("✓ 초기 자세 이동 완료")
            
            # 그리퍼 초기화 (17_teleop: 초기화 실패 시 종료)
            self.gripper = Gripper()
            if not self.gripper.initialize():
                # 17_teleop과 동일: 그리퍼 초기화 실패 시 정리 후 예외 발생
                logging.error("그리퍼 초기화 실패 - 안전을 위해 텔레오퍼레이션 중단")
                self.gripper = None
                # 이미 시작된 리소스 정리
                if self.command_stream:
                    try:
                        self.robot.cancel_control()
                        self.robot.disable_control_manager()
                        self.robot.power_off("12v")
                    except:
                        pass
                raise RuntimeError("그리퍼 초기화 실패 - 텔레오퍼레이션 불가")
            
            self.gripper.homing()
            self.gripper.start()
            
            # 마스터 암 초기화
            rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
            
            # ⚠️ 시작 시 온도 스냅샷 비활성화
            # DynamixelBus를 열면 버스가 제대로 해제되지 않아 MasterArm.initialize() 실패
            # 토크 데이터는 실시간으로 로깅됨, 온도는 종료 시에만 확인 가능
            
            # 마스터 암 URDF 경로 (17_teleop과 동일한 방식: 스크립트 상대경로 우선)
            # 1. 워크스페이스 내 rby1-sdk 경로
            sdk_path = Path(__file__).parent.parent / "rby1-sdk"
            if not sdk_path.exists():
                # 2. 스크립트와 같은 레벨의 rby1-sdk
                sdk_path = Path(__file__).parent / "rby1-sdk"
            if not sdk_path.exists():
                # 3. 홈 디렉토리 기본 경로
                sdk_path = Path.home() / "vla_ws" / "rby1-sdk"
            if not sdk_path.exists():
                sdk_path = Path.home() / "molmo_ws" / "rby1-sdk"
            
            master_arm_model = str(sdk_path / "models" / "master_arm" / "model.urdf")
            if not Path(master_arm_model).exists():
                raise FileNotFoundError(f"마스터 암 URDF 파일 없음: {master_arm_model}")
            
            self.master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
            self.master_arm.set_model_path(master_arm_model)
            self.master_arm.set_control_period(TeleopSettings.master_arm_loop_period)
            
            active_ids = self.master_arm.initialize(verbose=False)
            if len(active_ids) != rby.upc.MasterArm.DeviceCount:
                raise RuntimeError(f"마스터 암 장치 수 불일치 ({len(active_ids)}/{rby.upc.MasterArm.DeviceCount})")
            
            # 초기 목표 위치 설정
            self.right_q = None
            self.left_q = None
            self.right_minimum_time = 1.0
            self.left_minimum_time = 1.0
            
            # 마스터 암 제어 루프 시작
            self.master_arm.start_control(self._master_arm_control_loop)
            print("✓ 텔레오퍼레이션 준비 완료 (마스터 암 버튼으로 제어)")
            print("   → 버튼 누르면 해당 팔 제어 활성화")
            
        except AttributeError as e:
            print(f"⚠ 텔레오퍼레이션 설정 실패: UPC 기능 없음 ({e})")
            print("  → 이 기능은 UPC(Ubuntu PC)에서만 사용 가능합니다.")
            self._cleanup_teleop_on_error()
            raise RuntimeError(f"UPC 기능 없음: {e}")
        except Exception as e:
            print(f"⚠ 텔레오퍼레이션 설정 실패: {e}")
            self._cleanup_teleop_on_error()
            raise
    
    def _cleanup_teleop_on_error(self):
        """텔레오퍼레이션 설정 실패 시 안전 정리"""
        # 그리퍼 정리
        if self.gripper is not None:
            try:
                self.gripper.stop()
            except:
                pass
            self.gripper = None
        
        # 마스터 암 정리
        if self.master_arm is not None:
            try:
                self.master_arm.stop_control()
            except:
                pass
            self.master_arm = None
        
        # 로봇 제어 정리
        if self.command_stream is not None:
            try:
                self.robot.cancel_control()
                time.sleep(0.5)
                self.robot.disable_control_manager()
                self.robot.power_off("12v")
            except:
                pass
            self.command_stream = None
    
    def _move_j(self, pose: dict, minimum_time: float = 5.0) -> bool:
        """초기 자세로 이동 (공식 17_teleop의 move_j와 동일)
        
        블로킹 호출, Position 모드 사용 (Impedance 설정과 무관)
        SDK의 handler.get()이 완료될 때까지 대기
        """
        # Position 모드 빌더 (17_teleop 기본값)
        torso_builder = (
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0))
            .set_position(pose["torso"])
            .set_minimum_time(minimum_time)
        )
        
        right_arm_builder = (
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0))
            .set_position(pose["right_arm"])
            .set_minimum_time(minimum_time)
        )
        
        left_arm_builder = (
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0))
            .set_position(pose["left_arm"])
            .set_minimum_time(minimum_time)
        )
        
        cmd = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder()
                .set_torso_command(torso_builder)
                .set_right_arm_command(right_arm_builder)
                .set_left_arm_command(left_arm_builder)
            )
        )
        
        handler = self.robot.send_command(cmd)
        result = handler.get()
        return result == rby.RobotCommandFeedback.FinishCode.Ok

    def _send_ready_pose_stream(self, pose: dict, minimum_time: float = 5.0):
        """초기 자세로 이동 (command_stream 사용, 비블로킹)"""
        if self.command_stream is None:
            return
        
        # Position 또는 Impedance 모드에 따른 빌더 선택
        torso_builder = (
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
            .set_position(pose["torso"])
            .set_minimum_time(minimum_time)
        )
        
        # 오른팔 빌더
        right_arm_builder = (
            rby.JointPositionCommandBuilder()
            if self.position_mode
            else rby.JointImpedanceControlCommandBuilder()
        )
        (
            right_arm_builder
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
            .set_position(pose["right_arm"])
            .set_minimum_time(minimum_time)
        )
        if not self.position_mode:
            (
                right_arm_builder
                .set_stiffness([TeleopSettings.impedance_stiffness] * 7)
                .set_damping_ratio(TeleopSettings.impedance_damping_ratio)
                .set_torque_limit([TeleopSettings.impedance_torque_limit] * 7)
            )
        
        # 왼팔 빌더
        left_arm_builder = (
            rby.JointPositionCommandBuilder()
            if self.position_mode
            else rby.JointImpedanceControlCommandBuilder()
        )
        (
            left_arm_builder
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
            .set_position(pose["left_arm"])
            .set_minimum_time(minimum_time)
        )
        if not self.position_mode:
            (
                left_arm_builder
                .set_stiffness([TeleopSettings.impedance_stiffness] * 7)
                .set_damping_ratio(TeleopSettings.impedance_damping_ratio)
                .set_torque_limit([TeleopSettings.impedance_torque_limit] * 7)
            )
        
        cmd = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder()
                .set_torso_command(torso_builder)
                .set_right_arm_command(right_arm_builder)
                .set_left_arm_command(left_arm_builder)
            )
        )
        
        self.command_stream.send_command(cmd)
    
    def _wait_for_pose_reached(self, target_pose: dict, tolerance: float = 0.1, timeout: float = 10.0) -> bool:
        """목표 자세에 도달할 때까지 폴링 대기
        
        Args:
            target_pose: 목표 자세 dict (right_arm, left_arm 키)
            tolerance: 허용 오차 (라디안)
            timeout: 최대 대기 시간 (초)
            
        Returns:
            True if reached, False if timeout
        """
        start = time.time()
        right_target = np.array(target_pose["right_arm"])
        left_target = np.array(target_pose["left_arm"])
        
        while time.time() - start < timeout:
            with self.state_lock:
                if self.latest_state is None:
                    time.sleep(0.1)
                    continue
                current = np.array(self.latest_state.position)
            
            # 현재 관절 위치와 목표 비교
            right_current = current[self.robot_model.right_arm_idx]
            left_current = current[self.robot_model.left_arm_idx]
            
            right_error = np.max(np.abs(right_target - right_current))
            left_error = np.max(np.abs(left_target - left_current))
            
            elapsed = time.time() - start
            print(f"\r   대기 중... R_err:{right_error:.3f} L_err:{left_error:.3f} ({elapsed:.1f}s)", end="", flush=True)
            
            if right_error < tolerance and left_error < tolerance:
                print()  # 줄바꿈
                return True
            
            time.sleep(0.1)
        
        print()  # 줄바꿈
        return False
    
    def move_to_ready_pose(self, timeout: float = 5.0):
        """초기 자세로 이동 (에피소드 시작시 호출)
        
        Args:
            timeout: 최대 대기 시간 (초)
        """
        if not self.use_teleop or self.command_stream is None:
            return
        
        model_name = self.robot_model.model_name if self.robot_model else "A"
        ready_pose = READY_POSE.get(model_name, READY_POSE["A"])
        
        print("\n🔄 초기 자세로 이동 중...")
        
        # 공식 17_teleop처럼 blocking으로 이동 (Position 모드)
        if not self._move_j(ready_pose, minimum_time=2.0):
            print("   ⚠ 이동 실패 - 현재 위치에서 진행")
        
        # 마스터 암 목표 위치도 초기화
        if self.master_arm is not None:
            self.right_q = ready_pose["right_arm"].copy()
            self.left_q = ready_pose["left_arm"].copy()
            self.right_minimum_time = 1.0
            self.left_minimum_time = 1.0
        
        print("✓ 초기 자세 완료")
    
    def move_head(self, direction: str):
        """헤드 이동 (키보드 제어) - 즉시 명령 전송
        
        Args:
            direction: 'up', 'down', 'left', 'right', 'center'
        """
        # 방향에 따라 헤드 위치 업데이트
        if direction == 'up':
            self.head_q[1] -= self.head_step  # tilt 감소 (위로)
        elif direction == 'down':
            self.head_q[1] += self.head_step  # tilt 증가 (아래로)
        elif direction == 'left':
            self.head_q[0] += self.head_step  # pan 증가 (왼쪽)
        elif direction == 'right':
            self.head_q[0] -= self.head_step  # pan 감소 (오른쪽)
        elif direction == 'center':
            self.head_q = np.array([0.0, 0.0])  # 중앙으로 리셋
        
        # 제한 범위 적용
        self.head_q[0] = np.clip(self.head_q[0], *self.head_limits['pan'])
        self.head_q[1] = np.clip(self.head_q[1], *self.head_limits['tilt'])
        # 마스터 암 루프에서 100Hz로 head_q 값을 command_stream으로 전송함
    
    def _move_to_ready_pose(self, pose: dict, minimum_time: float = 5.0, timeout: float = 10.0) -> bool:
        """초기 자세로 이동 (타임아웃 포함)"""
        try:
            # Joint position command 빌더
            torso_builder = (
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                .set_position(pose["torso"])
                .set_minimum_time(minimum_time)
            )
            right_arm_builder = (
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                .set_position(pose["right_arm"])
                .set_minimum_time(minimum_time)
            )
            left_arm_builder = (
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                .set_position(pose["left_arm"])
                .set_minimum_time(minimum_time)
            )
            
            cmd = rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_torso_command(torso_builder)
                    .set_right_arm_command(right_arm_builder)
                    .set_left_arm_command(left_arm_builder)
                )
            )
            
            handler = self.robot.send_command(cmd)
            
            # 타임아웃 적용하여 대기
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(handler.get)
                try:
                    result = future.result(timeout=timeout)
                    return result == rby.RobotCommandFeedback.FinishCode.Ok
                except concurrent.futures.TimeoutError:
                    print(f"⚠ 초기 자세 이동 타임아웃 ({timeout}초) - 현재 위치에서 시작")
                    return False
        except Exception as e:
            print(f"⚠ 초기 자세 이동 오류: {e}")
            return False
    
    # _check_safety_limits 제거됨 - 17_teleop과 동일하게 별도 안전 모니터링 없이 동작
    # 마스터 암 상태에는 temperature/current/torque 필드가 없음
    # 로봇 본체는 RBY1 SDK 내부에서 자체 안전 관리됨
    
    def _master_arm_control_loop(self, state):
        """마스터 암 제어 콜백 - 로봇을 실시간으로 제어

        
        Note: 이 콜백은 100Hz로 호출되므로 예외 발생 시 안전하게 처리해야 함
        """
        try:
            return self._master_arm_control_loop_inner(state)
        except Exception as e:
            # 콜백 내부 예외 발생 시 안전하게 처리
            logging.error(f"마스터 암 콜백 오류: {e}")
            # 기본 입력 반환 (로봇 정지 상태 유지)
            return rby.upc.MasterArm.ControlInput()
    
    def _master_arm_control_loop_inner(self, state):
        """마스터 암 제어 콜백 내부 구현
        
        Args:
            state: 마스터 암 상태 (rby.upc.MasterArm.State)
                   - q_joint: 마스터 암 관절 위치
                   - qvel_joint: 마스터 암 관절 속도
                   - button_right/left: 버튼 상태
                   - gravity_term: 중력 보상 토크
        """
        # 현재 마스터 암 상태 저장 (녹화용)
        with self.master_arm_lock:
            self.master_arm_state = state
        
        # ========================================================================
        # 마스터 암 토크 모니터링 (state.torque_joint 사용)
        # ========================================================================
        if hasattr(state, 'torque_joint') and state.torque_joint is not None:
            ma_torques = np.abs(np.array(state.torque_joint))
            
            # Critical 체크
            over_critical = ma_torques > TeleopSettings.ma_torque_critical
            if np.any(over_critical):
                critical_idx = np.where(over_critical)[0][0]
                if not self._teleop_paused:
                    self._teleop_paused = True
                    arm_name = "Right" if critical_idx < 7 else "Left"
                    joint_idx = critical_idx if critical_idx < 7 else critical_idx - 7
                    self._critical_reason = f"마스터암 토크 과부하! {arm_name} joint {joint_idx}: {ma_torques[critical_idx]:.2f}Nm (limit: {TeleopSettings.ma_torque_critical[critical_idx]:.2f}Nm)"
                    logging.critical(self._critical_reason)
                    self._write_safety_log("CRITICAL", self._critical_reason)
                    # 마스터암 해제 요청 (콜백 내부에서 직접 stop_control 호출하면 데드락)
                    self._ma_disconnect_requested = True
                    # 별도 스레드에서 안전하게 해제
                    import threading
                    threading.Thread(target=self._disconnect_master_arm_safe, daemon=True).start()
            else:
                # Warning 체크 (1초에 1회만 로그)
                over_warning = ma_torques > TeleopSettings.ma_torque_warning
                if np.any(over_warning):
                    self._ma_warning_count += 1
                    if self._ma_warning_count >= round(1 / TeleopSettings.master_arm_loop_period):
                        warning_idx = np.where(over_warning)[0][0]
                        warning_msg = f"마스터암 토크 경고: 관절 {warning_idx}: {ma_torques[warning_idx]:.2f}Nm"
                        logging.warning(warning_msg)
                        self._write_safety_log("WARNING", warning_msg)
                        self._ma_warning_count = 0
                else:
                    self._ma_warning_count = 0
        
        # Teleop 일시정지 상태면 기본 입력만 반환 (로봇 위치 유지)
        if self._teleop_paused:
            return rby.upc.MasterArm.ControlInput()
        
        # 로그 파일에 저장 (17_teleop과 동일: 매초 버튼/트리거 상태)
        self._ma_log_count += 1
        if self._ma_log_count % round(1 / TeleopSettings.master_arm_loop_period) == 0:
            self._write_teleop_log(state)
            self._ma_log_count = 0
        
        # 로봇 관절 위치가 없으면 대기
        if self.robot_q is None:
            with self.state_lock:
                if self.latest_state is not None:
                    self.robot_q = np.array(self.latest_state.position)
            return rby.upc.MasterArm.ControlInput()
        
        # 초기 목표 위치 설정
        if self.right_q is None:
            self.right_q = np.array(state.q_joint[0:7])
        if self.left_q is None:
            self.left_q = np.array(state.q_joint[7:14])
        
        ma_input = rby.upc.MasterArm.ControlInput()
        
        # 그리퍼 제어
        if self.gripper:
            self.gripper.set_target(np.array([
                state.button_right.trigger / 1000.0,
                state.button_left.trigger / 1000.0
            ]))
        
        # 마스터 암 토크 계산
        torque = (
            state.gravity_term
            + MA_Q_LIMIT_BARRIER * (
                np.maximum(MA_MIN_Q - state.q_joint, 0)
                + np.minimum(MA_MAX_Q - state.q_joint, 0)
            )
            + MA_VISCOUS_GAIN * state.qvel_joint
        )
        torque = np.clip(torque, -MA_TORQUE_LIMIT, MA_TORQUE_LIMIT)
        
        # 오른팔 마스터 암 제어 (토크 게인 0.6 - 17_teleop 기본값과 동일)
        if state.button_right.button == 1:
            ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentControlMode)
            ma_input.target_torque[0:7] = torque[0:7] * 0.4  # 17_teleop 기본값: 0.6
            self.right_q = np.array(state.q_joint[0:7])
        else:
            ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            ma_input.target_torque[0:7] = MA_TORQUE_LIMIT[0:7]
            ma_input.target_position[0:7] = self.right_q
        
        # 왼팔 마스터 암 제어 (토크 게인 0.6 - 17_teleop 기본값과 동일)
        if state.button_left.button == 1:
            ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentControlMode)
            ma_input.target_torque[7:14] = torque[7:14] * 0.4  # 17_teleop 기본값: 0.6
            self.left_q = np.array(state.q_joint[7:14])
        else:
            ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            ma_input.target_torque[7:14] = MA_TORQUE_LIMIT[7:14]
            ma_input.target_position[7:14] = self.left_q
        
        # 충돌 체크
        q = self.robot_q.copy()
        q[self.robot_model.right_arm_idx] = self.right_q
        q[self.robot_model.left_arm_idx] = self.left_q
        self.dyn_state.set_q(q)
        self.dyn_robot.compute_forward_kinematics(self.dyn_state)
        is_collision = self.dyn_robot.detect_collisions_or_nearest_links(self.dyn_state, 1)[0].distance < 0.02
        
        # 로봇 명령 빌드
        rc = rby.BodyComponentBasedCommandBuilder()
        
        if state.button_right.button and not is_collision:
            self.right_minimum_time -= TeleopSettings.master_arm_loop_period
            self.right_minimum_time = max(self.right_minimum_time, TeleopSettings.master_arm_loop_period * 1.01)
            
            # Position 또는 Impedance 모드 선택
            right_arm_builder = (
                rby.JointPositionCommandBuilder()
                if self.position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                right_arm_builder
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                .set_position(np.clip(self.right_q, self.robot_min_q[self.robot_model.right_arm_idx], 
                                      self.robot_max_q[self.robot_model.right_arm_idx]))
                .set_velocity_limit(self.robot_max_qdot[self.robot_model.right_arm_idx])
                .set_acceleration_limit(self.robot_max_qddot[self.robot_model.right_arm_idx] * 30)
                .set_minimum_time(self.right_minimum_time)
            )
            # Impedance 모드 추가 설정
            if not self.position_mode:
                (
                    right_arm_builder
                    .set_stiffness([TeleopSettings.impedance_stiffness] * len(self.robot_model.right_arm_idx))
                    .set_damping_ratio(TeleopSettings.impedance_damping_ratio)
                    .set_torque_limit([TeleopSettings.impedance_torque_limit] * len(self.robot_model.right_arm_idx))
                )
            rc.set_right_arm_command(right_arm_builder)
        else:
            self.right_minimum_time = 0.8
        
        if state.button_left.button and not is_collision:
            self.left_minimum_time -= TeleopSettings.master_arm_loop_period
            self.left_minimum_time = max(self.left_minimum_time, TeleopSettings.master_arm_loop_period * 1.01)
            
            # Position 또는 Impedance 모드 선택
            left_arm_builder = (
                rby.JointPositionCommandBuilder()
                if self.position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                left_arm_builder
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1e6))
                .set_position(np.clip(self.left_q, self.robot_min_q[self.robot_model.left_arm_idx],
                                      self.robot_max_q[self.robot_model.left_arm_idx]))
                .set_velocity_limit(self.robot_max_qdot[self.robot_model.left_arm_idx])
                .set_acceleration_limit(self.robot_max_qddot[self.robot_model.left_arm_idx] * 30)
                .set_minimum_time(self.left_minimum_time)
            )
            # Impedance 모드 추가 설정
            if not self.position_mode:
                (
                    left_arm_builder
                    .set_stiffness([TeleopSettings.impedance_stiffness] * len(self.robot_model.left_arm_idx))
                    .set_damping_ratio(TeleopSettings.impedance_damping_ratio)
                    .set_torque_limit([TeleopSettings.impedance_torque_limit] * len(self.robot_model.left_arm_idx))
                )
            rc.set_left_arm_command(left_arm_builder)
        else:
            self.left_minimum_time = 0.8
        
        # 로봇에 명령 전송 (body만 - 마스터 암 버튼 눌렀을 때만)
        if self.command_stream:
            try:
                has_arm_command = state.button_right.button or state.button_left.button
                
                if has_arm_command:
                    cmd_builder = rby.ComponentBasedCommandBuilder().set_body_command(rc)
                    self.command_stream.send_command(
                        rby.RobotCommandBuilder().set_command(cmd_builder)
                    )
            except RuntimeError as e:
                # command_stream 만료시 재생성
                if "expired" in str(e):
                    try:
                        self.command_stream = self.robot.create_command_stream(priority=1)
                    except:
                        pass
        
        return ma_input

    def get_master_arm_action(self) -> dict | None:
        """마스터 암에서 action 값 가져오기"""
        if self.master_arm is None:
            return None
        
        with self.master_arm_lock:
            state = self.master_arm_state
        
        if state is None:
            return None
        
        action = {}
        
        # 마스터 암 관절 위치: state.q_joint
        # 오른팔: [0:7], 왼팔: [7:14]
        ma_joints = np.array(state.q_joint)
        
        if self.arms == "right":
            # 오른팔만
            for i, name in enumerate(self.joint_names):
                action[f"{name}.pos"] = float(ma_joints[i]) if i < 7 else 0.0
            # 그리퍼: 트리거 값 (0-1000 -> 0-1 정규화)
            action["right_gripper.pos"] = float(state.button_right.trigger) / 1000.0
            
        elif self.arms == "left":
            # 왼팔만
            for i, name in enumerate(self.joint_names):
                action[f"{name}.pos"] = float(ma_joints[7 + i]) if i < 7 else 0.0
            action["left_gripper.pos"] = float(state.button_left.trigger) / 1000.0
            
        else:  # both
            # 양팔
            for i, name in enumerate(self.joint_names):
                if i < 7:  # 오른팔
                    action[f"{name}.pos"] = float(ma_joints[i])
                else:  # 왼팔
                    action[f"{name}.pos"] = float(ma_joints[i])  # 7:14
            action["right_gripper.pos"] = float(state.button_right.trigger) / 1000.0
            action["left_gripper.pos"] = float(state.button_left.trigger) / 1000.0
        
        return action

    def disconnect(self):
        """연결 해제
        
        종료 순서 (공식 17_teleoperation_with_joint_mapping.py 기준):
          1. stop_state_update   ← 상태 콜백 먼저 중지
          2. master_arm.stop_control()
          3. cancel_control
          4. sleep(0.5)          ← 명령 완료 대기
          5. disable_control_manager
          6. power_off
          7. gripper.stop()      ← 그리퍼는 마지막
        """
        # 중복 호출 방지
        if hasattr(self, '_disconnected') and self._disconnected:
            return
        self._disconnected = True
        
        # 시그널 핸들러 복원
        self._restore_signal_handlers()
        
        # 로그 파일 닫기
        self._close_log_file()
        if self._log_dir:
            print(f"✓ 로그 저장 완료: {self._log_dir}")
            self._log_dir = None  # 중복 출력 방지
        
        # === 공식 순서 적용 ===
        
        # 1. 상태 업데이트 중지 (중복 호출 방지) ← 먼저!
        if self.robot and not self._state_update_stopped:
            try:
                self.robot.stop_state_update()
                self._state_update_stopped = True
                print("✓ 상태 업데이트 중지")
            except Exception:
                pass
        
        # 2. 마스터 암 해제 (중복 호출 방지)
        if self.master_arm is not None and not self._master_arm_stopped:
            try:
                self.master_arm.stop_control()
                self._master_arm_stopped = True
                print("✓ 마스터 암 연결 해제")
                # 종료 후 온도/전류 스냅샷 (다이나믹셀 문제 분석용)
                self._read_master_arm_motor_states("END")
            except Exception:
                pass
        
        # 3-6. 텔레오퍼레이션 모드: 제어권 해제 (중복 호출 방지)
        if self.use_teleop and self.robot:
            try:
                # 3. cancel_control
                if not self._robot_control_cancelled:
                    self.robot.cancel_control()
                    self._robot_control_cancelled = True
                # 4. sleep (공식: 0.5초)
                time.sleep(0.5)
                # 5. disable_control_manager
                self.robot.disable_control_manager()
                # 6. power_off
                self.robot.power_off("12v")
                print("✓ 제어권 해제")
            except Exception:
                pass
        
        # 7. 그리퍼 해제 ← 마지막!
        if self.gripper is not None:
            try:
                self.gripper.stop()
                print("✓ 그리퍼 연결 해제")
            except Exception:
                pass
        
        print("✓ 로봇 연결 해제 완료")

        # 멀티 RealSense 카메라 해제
        if self.rs_pipelines:
            for cam_name, (pipeline, _) in self.rs_pipelines.items():
                try:
                    pipeline.stop()
                except Exception:
                    pass
            print(f"✓ {len(self.rs_pipelines)}개 RealSense 카메라 연결 해제")
            self.rs_pipelines = {}
            self.rs_pipeline = None

        if self.camera:
            self.camera.release()
            print("✓ 카메라 연결 해제")

    def get_observation(self) -> dict:
        """현재 관측 데이터 수집"""
        obs = {}

        # 로봇 상태
        with self.state_lock:
            state = self.latest_state

        if state is not None:
            positions = np.array(state.position)
            velocities = np.array(state.velocity)
            torques = np.array(state.torque)

            # 선택한 팔의 관절 인덱스 가져오기
            if self.arms == "right":
                joint_indices = list(self.robot_model.right_arm_idx) if self.robot_model else list(range(6, 13))
            elif self.arms == "left":
                joint_indices = list(self.robot_model.left_arm_idx) if self.robot_model else list(range(13, 20))
            else:  # both
                right_idx = list(self.robot_model.right_arm_idx) if self.robot_model else list(range(6, 13))
                left_idx = list(self.robot_model.left_arm_idx) if self.robot_model else list(range(13, 20))
                joint_indices = right_idx + left_idx

            for i, name in enumerate(self.joint_names):
                if i < len(joint_indices):
                    idx = joint_indices[i]
                    if idx < len(positions):
                        obs[f"{name}.pos"] = float(positions[idx])
                        obs[f"{name}.vel"] = float(velocities[idx])
                        obs[f"{name}.torque"] = float(torques[idx])

            # [개발중] 휠 데이터 수집
            if self.use_wheels and self.robot_model is not None:
                try:
                    # 휠 인덱스 가져오기 (RBY1-A: wheel_0=22, wheel_1=23)
                    wheel_indices = getattr(self.robot_model, 'wheel_idx', None)
                    if wheel_indices is None:
                        # 기본 인덱스 사용 (head 다음)
                        wheel_indices = [22, 23]
                    
                    for i, wheel_name in enumerate(WHEEL_JOINTS):
                        if i < len(wheel_indices):
                            idx = wheel_indices[i]
                            if idx < len(positions):
                                obs[f"{wheel_name}.pos"] = float(positions[idx])
                                obs[f"{wheel_name}.vel"] = float(velocities[idx])
                                obs[f"{wheel_name}.torque"] = float(torques[idx])
                except Exception as e:
                    # 휠 데이터 수집 실패시 0으로 채움
                    for wheel_name in WHEEL_JOINTS:
                        obs[f"{wheel_name}.pos"] = 0.0
                        obs[f"{wheel_name}.vel"] = 0.0
                        obs[f"{wheel_name}.torque"] = 0.0

            # 그리퍼 상태 (tool_state에서 가져오기)
            try:
                if hasattr(state, 'tool_state') and state.tool_state is not None:
                    tool = state.tool_state
                    if self.arms in ["right", "both"]:
                        if hasattr(tool, 'right_gripper_position'):
                            obs["right_gripper.pos"] = float(tool.right_gripper_position)
                        elif hasattr(tool, 'right_tool_position'):
                            obs["right_gripper.pos"] = float(tool.right_tool_position)
                        else:
                            obs["right_gripper.pos"] = 0.0
                    if self.arms in ["left", "both"]:
                        if hasattr(tool, 'left_gripper_position'):
                            obs["left_gripper.pos"] = float(tool.left_gripper_position)
                        elif hasattr(tool, 'left_tool_position'):
                            obs["left_gripper.pos"] = float(tool.left_tool_position)
                        else:
                            obs["left_gripper.pos"] = 0.0
                else:
                    if self.arms in ["right", "both"]:
                        obs["right_gripper.pos"] = 0.0
                    if self.arms in ["left", "both"]:
                        obs["left_gripper.pos"] = 0.0
            except Exception:
                if self.arms in ["right", "both"]:
                    obs["right_gripper.pos"] = 0.0
                if self.arms in ["left", "both"]:
                    obs["left_gripper.pos"] = 0.0

            # EEF pose 계산
            if self.dyn_robot is not None:
                try:
                    self._compute_eef_pose(positions, obs)
                except Exception as e:
                    pass  # EEF 계산 실패시 무시

        # 멀티 RealSense 카메라 이미지
        if self.rs_pipelines:
            # 스트리밍 중이면 stream_frames에서 가져오기 (레이스 컨디션 방지)
            if self.stream_port > 0 and hasattr(self, '_stream_capture_threads') and self._stream_capture_threads:
                with self.stream_lock:
                    for cam_name in self.rs_pipelines.keys():
                        if cam_name in self.stream_frames:
                            obs[cam_name] = self.stream_frames[cam_name].copy()
                        else:
                            obs[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # 스트리밍 안 할 때는 직접 캡처
                try:
                    import pyrealsense2 as rs
                    for cam_name, (pipeline, _) in self.rs_pipelines.items():
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=100)
                            color_frame = frames.get_color_frame()
                            if color_frame:
                                frame_rgb = np.asanyarray(color_frame.get_data())
                                obs[cam_name] = frame_rgb
                            else:
                                obs[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
                        except Exception:
                            obs[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
                except Exception as e:
                    pass  # 전체 카메라 실패시 무시
        elif self.camera is not None:
            # 일반 USB 카메라
            import cv2
            ret, frame = self.camera.read()
            if ret:
                # BGR -> RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs["camera"] = frame_rgb
                # 웹 스트리밍용 버퍼에 저장
                if self.stream_port > 0:
                    with self.stream_lock:
                        self.stream_frames["camera"] = frame_rgb.copy()

        return obs

    def _compute_eef_pose(self, q: np.ndarray, obs: dict):
        """EEF pose 및 delta pose 계산"""
        # 관절 각도 설정
        self.dyn_state.set_q(q)
        
        # Forward kinematics 계산
        self.dyn_robot.compute_forward_kinematics(self.dyn_state)
        
        # 각 팔의 EEF pose 추출
        for arm, eef_name, link_idx in [("right", "ee_right", 1), ("left", "ee_left", 2)]:
            if arm == "right" and self.arms not in ["right", "both"]:
                continue
            if arm == "left" and self.arms not in ["left", "both"]:
                continue
            if arm == "left" and self.arms == "right":
                continue
            
            # base에서 EEF까지의 변환 행렬 계산
            actual_link_idx = 1 if (self.arms == "right" or (self.arms == "both" and arm == "right")) else 1
            if self.arms == "both" and arm == "left":
                actual_link_idx = 2
            elif self.arms == "left":
                actual_link_idx = 1
                
            try:
                T = self.dyn_robot.compute_transformation(self.dyn_state, 0, actual_link_idx)
                
                # Position (x, y, z)
                pos = T[:3, 3]
                
                # Rotation matrix to euler angles (roll, pitch, yaw)
                rot = T[:3, :3]
                euler = self._rotation_matrix_to_euler(rot)
                
                # 현재 pose 저장
                current_pose = np.concatenate([pos, euler])
                
                # Delta pose 계산
                prev_key = f"{arm}_eef"
                if prev_key in self.prev_eef_pose:
                    delta_pose = current_pose - self.prev_eef_pose[prev_key]
                else:
                    delta_pose = np.zeros(6)
                
                self.prev_eef_pose[prev_key] = current_pose.copy()
                
                # obs에 저장
                obs[f"{arm}_eef.pos_x"] = float(pos[0])
                obs[f"{arm}_eef.pos_y"] = float(pos[1])
                obs[f"{arm}_eef.pos_z"] = float(pos[2])
                obs[f"{arm}_eef.rot_roll"] = float(euler[0])
                obs[f"{arm}_eef.rot_pitch"] = float(euler[1])
                obs[f"{arm}_eef.rot_yaw"] = float(euler[2])
                obs[f"{arm}_eef.delta_x"] = float(delta_pose[0])
                obs[f"{arm}_eef.delta_y"] = float(delta_pose[1])
                obs[f"{arm}_eef.delta_z"] = float(delta_pose[2])
                obs[f"{arm}_eef.delta_roll"] = float(delta_pose[3])
                obs[f"{arm}_eef.delta_pitch"] = float(delta_pose[4])
                obs[f"{arm}_eef.delta_yaw"] = float(delta_pose[5])
            except Exception:
                pass

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Rotation matrix to euler angles (roll, pitch, yaw)"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])

    def _get_state_dim(self) -> int:
        """observation.state 벡터 차원 계산"""
        # 관절 수 + 그리퍼 수
        dim = len(self.joint_names)  # 관절 위치
        if self.arms in ["right", "both"]:
            dim += 1  # right gripper
        if self.arms in ["left", "both"]:
            dim += 1  # left gripper
        return dim

    def _get_state_names(self) -> list[str]:
        """observation.state 벡터의 각 요소 이름"""
        names = [f"{name}.pos" for name in self.joint_names]
        if self.arms in ["right", "both"]:
            names.append("right_gripper.pos")
        if self.arms in ["left", "both"]:
            names.append("left_gripper.pos")
        return names

    def build_features(self, use_camera: bool = False, camera_shape: tuple = (480, 640, 3)) -> dict:
        """데이터셋 feature 정의 생성 (LeRobot 표준 형식)"""
        features = {}
        
        state_dim = self._get_state_dim()
        state_names = self._get_state_names()

        # ===== LeRobot 표준 형식 =====
        # observation.state: 모든 관절+그리퍼 위치를 단일 벡터로
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"motors": state_names},
        }
        
        # action: 목표 관절+그리퍼 위치 벡터
        features["action"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"motors": state_names},
        }

        # ===== 추가 정보 (선택적) =====
        # 속도 벡터
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(self.joint_names),),
            "names": {"motors": [f"{name}.vel" for name in self.joint_names]},
        }
        
        # 토크 벡터
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(self.joint_names),),
            "names": {"motors": [f"{name}.torque" for name in self.joint_names]},
        }

        # EEF pose (dynamics 모델 사용 가능시)
        if self.dyn_robot is not None:
            eef_dim = 0
            eef_names = []
            if self.arms in ["right", "both"]:
                eef_dim += 6
                eef_names.extend(["right_eef.x", "right_eef.y", "right_eef.z", 
                                  "right_eef.roll", "right_eef.pitch", "right_eef.yaw"])
            if self.arms in ["left", "both"]:
                eef_dim += 6
                eef_names.extend(["left_eef.x", "left_eef.y", "left_eef.z",
                                  "left_eef.roll", "left_eef.pitch", "left_eef.yaw"])
            
            features["observation.eef_pos"] = {
                "dtype": "float32",
                "shape": (eef_dim,),
                "names": {"coords": eef_names},
            }

        # 카메라 (멀티 카메라 지원)
        if use_camera:
            if self.rs_pipelines:
                # 멀티 RealSense 카메라
                for cam_name in self.rs_pipelines.keys():
                    features[f"observation.images.{cam_name}"] = {
                        "dtype": "video",
                        "shape": camera_shape,
                        "names": ["height", "width", "channels"],
                    }
            elif self.camera is not None:
                # 단일 USB 카메라
                features["observation.images.camera"] = {
                    "dtype": "video",
                    "shape": camera_shape,
                    "names": ["height", "width", "channels"],
                }

        return features

    def record_episodes(
        self,
        output_name: str,
        task: str,
        num_episodes: int = 1,
        fps: int = 30,
        use_camera: bool = False,
    ):
        """여러 에피소드 녹화 (키보드 제어)"""
        print("\n" + "=" * 60)
        print(f"녹화 설정")
        print("=" * 60)
        print(f"  출력: {output_name}")
        print(f"  태스크: {task}")
        print(f"  에피소드 수: {num_episodes}")
        print(f"  FPS: {fps}")
        print(f"  최대 에피소드 시간: {MAX_EPISODE_DURATION}초")
        print(f"  팔 선택: {self.arms} ({len(self.joint_names)}개 관절)")
        
        # 텔레오퍼레이션 모드 상태
        if self.use_teleop:
            if self.master_arm is not None:
                teleop_status = "✓ 활성화 (마스터 암 연결됨)"
            else:
                teleop_status = "⚠ 요청됨 (마스터 암 연결 실패 - 기본 모드로 동작)"
        else:
            teleop_status = "비활성화 (action = observation.state)"
        print(f"  텔레오퍼레이션: {teleop_status}")
        
        # 카메라 상태
        if use_camera and self.has_camera:
            if self.rs_pipelines:
                cam_names = list(self.rs_pipelines.keys())
                cam_status = f'RealSense {len(cam_names)}대 ({", ".join(cam_names)})'
            elif self.camera is not None:
                cam_status = 'USB 카메라 1대'
            else:
                cam_status = '비활성화'
        else:
            cam_status = '비활성화'
        print(f"  카메라: {cam_status}")
        
        # [개발중] 휠 기록 상태
        if self.use_wheels:
            wheel_status = "✓ 활성화 [개발중]"
        else:
            wheel_status = "비활성화"
        print(f"  휠 기록: {wheel_status}")
        
        # 초기 자세 리셋 상태
        if self.use_teleop and self.reset_pose_each_episode:
            reset_status = "✓ 활성화 (매 에피소드 시작시 초기 자세로 이동)"
        else:
            reset_status = "비활성화 (--no-reset 또는 teleop 비활성)"
        print(f"  초기 자세 리셋: {reset_status}")
        print(f"  제어 모드: {self.control_mode}")
        print("=" * 60)
        print("\n키보드 조작:")
        print("  [SPACE] 녹화 시작/일시정지")
        print("  [ENTER] 에피소드 저장 & 다음으로")
        print("  [R]     현재 에피소드 취소 & 다시 녹화")
        print("  [B]     이전 에피소드 삭제 & 재녹화")
        print("  [T]     Teleop 재연결 (Critical 해제 후)")
        print("  [Q]     종료")
        # 헤드 제어 기능 비활성화 (안전 문제로 제거됨)
        # if self.use_teleop:
        #     print("  ─────── 헤드 제어 ───────")
        #     print("  [W/S]   헤드 위/아래 (tilt)")
        #     print("  [A/D]   헤드 좌/우 (pan)")
        #     print("  [X]     헤드 중앙 리셋")
        print("=" * 60)

        # Feature 정의
        use_cam = use_camera and self.has_camera
        features = self.build_features(use_camera=use_cam)

        # 저장 경로 설정: ~/vla_ws/datasets/
        save_root = Path.home() / "vla_ws" / "datasets"
        save_root.mkdir(parents=True, exist_ok=True)

        # 데이터셋 생성
        dataset = LeRobotDataset.create(
            repo_id=f"local/{output_name}",
            fps=fps,
            root=save_root / output_name,
            robot_type="rby1",
            features=features,
            use_videos=use_cam,
        )
        print(f"\n데이터셋 경로: {dataset.root}")

        frame_interval = 1.0 / fps
        episode_idx = 0
        total_frames = 0
        episode_frame_counts = []  # 각 에피소드별 프레임 수 저장

        with KeyboardController() as keyboard:
            while episode_idx < num_episodes:
                print(f"\n{'='*60}")
                print(f"에피소드 {episode_idx + 1}/{num_episodes}")
                if episode_idx > 0:
                    print(f"(이전 에피소드 재녹화: [B] 키)")
                print(f"{'='*60}")
                print("SPACE를 눌러 녹화를 시작하세요...")

                # 녹화 시작 대기
                recording = False
                episode_done = False
                episode_cancelled = False
                frame_count = 0
                episode_start_time = None
                
                # 에피소드 시작시 이전 EEF pose 초기화
                self.prev_eef_pose = {}
                
                # 에피소드 시작시 초기 자세로 이동 (teleop 모드 + reset 활성화시)
                if self.use_teleop and self.reset_pose_each_episode:
                    self.move_to_ready_pose(timeout=3.0)

                while not episode_done:
                    key = keyboard.get_key(timeout=0.05)

                    if key:
                        if key == ' ':  # SPACE - 녹화 토글
                            recording = not recording
                            if recording:
                                if episode_start_time is None:
                                    episode_start_time = time.time()
                                print("\n▶ 녹화 시작!")
                            else:
                                print("\n⏸ 녹화 일시정지")

                        elif key == '\n' or key == '\r':  # ENTER - 에피소드 저장
                            if frame_count > 0:
                                episode_done = True
                                print("\n✓ 에피소드 저장 중...")
                            else:
                                print("\n⚠ 녹화된 프레임이 없습니다!")

                        elif key.lower() == 'r':  # R - 현재 에피소드 취소
                            if frame_count > 0:
                                episode_cancelled = True
                                episode_done = True
                                print("\n✗ 현재 에피소드 취소됨")
                            else:
                                print("\n취소할 녹화가 없습니다.")

                        elif key.lower() == 'b':  # B - 이전 에피소드 재녹화
                            if episode_idx > 0:
                                # 확인 절차
                                print(f"\n⚠ 에피소드 {episode_idx}을(를) 삭제하고 재녹화할까요? (y/n): ", end="", flush=True)
                                confirm_key = keyboard.get_key(timeout=10)
                                if confirm_key and confirm_key.lower() == 'y':
                                    # 현재 녹화 중인 데이터 취소
                                    if frame_count > 0:
                                        dataset.clear_episode_buffer()
                                        print(f"현재 에피소드 {episode_idx + 1} 버퍼 삭제됨")
                                    
                                    # 이전 에피소드 삭제
                                    try:
                                        dataset.delete_episode(episode_idx - 1)
                                        prev_frames = episode_frame_counts.pop()
                                        total_frames -= prev_frames
                                        episode_idx -= 1
                                        print(f"◀ 에피소드 {episode_idx + 1} 삭제됨 ({prev_frames} 프레임). 재녹화합니다...")
                                        episode_done = True
                                        episode_cancelled = True  # 현재 루프 종료, 다시 시작
                                    except Exception as e:
                                        print(f"\n⚠ 이전 에피소드 삭제 실패: {e}")
                                else:
                                    print("취소됨.")
                            else:
                                print("\n⚠ 첫 번째 에피소드입니다. 이전 에피소드가 없습니다.")

                        elif key.lower() == 'q':  # Q - 종료
                            print("\n종료합니다...")
                            if frame_count > 0:
                                # 현재 에피소드 저장 여부 확인
                                print("현재 에피소드를 저장할까요? (y/n): ", end="", flush=True)
                                save_key = keyboard.get_key(timeout=10)
                                if save_key and save_key.lower() == 'y':
                                    dataset.save_episode()
                                    episode_idx += 1
                                    total_frames += frame_count
                                else:
                                    dataset.clear_episode_buffer()
                            
                            # 최종 저장
                            if episode_idx > 0:
                                dataset.finalize()
                                self._print_summary(output_name, episode_idx, total_frames, save_root)
                            return dataset
                        
                        elif key.lower() == 't':  # T - Teleop 재연결
                            if self._teleop_paused:
                                print("\n🔄 Teleop 재연결 시도 중...")
                                self._teleop_paused = False
                                self._critical_reason = ""
                                print("✓ Teleop 재연결됨. 녹화를 계속하려면 SPACE를 누르세요.")
                            else:
                                print("\n⚠ Teleop이 이미 활성화 상태입니다.")
                        
                        # 헤드 제어 기능 비활성화 (안전 문제로 제거됨)
                        # elif self.use_teleop and self.command_stream is not None:
                        #     if key.lower() == 'w':  # W - 헤드 위로
                        #         self.move_head('up')
                        #     elif key.lower() == 's':  # S - 헤드 아래로
                        #         self.move_head('down')
                        #     elif key.lower() == 'a':  # A - 헤드 왼쪽
                        #         self.move_head('left')
                        #     elif key.lower() == 'd':  # D - 헤드 오른쪽
                        #         self.move_head('right')
                        #     elif key.lower() == 'x':  # X - 헤드 중앙으로
                        #         self.move_head('center')

                    # 녹화 중일 때 프레임 수집
                    if recording:
                        # Critical 상태 감지 시 녹화 일시정지
                        if self._teleop_paused and self.use_teleop:
                            recording = False
                            print(f"\n🔴 [CRITICAL] {self._critical_reason}")
                            print("   녹화 일시정지됨. 현재 에피소드 버퍼 유지.")
                            print("   → [T] Teleop 재연결 후 [R] 재녹화 또는 [SPACE] 계속")
                            continue
                        
                        loop_start = time.perf_counter()
                        elapsed = time.time() - episode_start_time

                        # 최대 시간 체크
                        if elapsed >= MAX_EPISODE_DURATION:
                            print(f"\n⏱ 최대 시간({MAX_EPISODE_DURATION}초) 도달! 에피소드 자동 저장...")
                            episode_done = True
                            continue

                        # 관측 수집
                        raw_obs = self.get_observation()

                        # 프레임 구성 (LeRobot 표준 형식)
                        frame = {"task": task}

                        # ===== observation.state: 관절+그리퍼 위치 벡터 =====
                        state_values = []
                        for name in self.joint_names:
                            state_values.append(raw_obs.get(f"{name}.pos", 0.0))
                        if self.arms in ["right", "both"]:
                            state_values.append(raw_obs.get("right_gripper.pos", 0.0))
                        if self.arms in ["left", "both"]:
                            state_values.append(raw_obs.get("left_gripper.pos", 0.0))
                        
                        frame["observation.state"] = np.array(state_values, dtype=np.float32)
                        
                        # ===== action: 목표 위치 벡터 =====
                        if self.use_teleop and self.master_arm is not None:
                            # 텔레오퍼레이션 모드: 마스터 암 위치를 action으로
                            ma_action = self.get_master_arm_action()
                            if ma_action is not None:
                                action_values = []
                                for name in self.joint_names:
                                    action_values.append(ma_action.get(f"{name}.pos", 0.0))
                                if self.arms in ["right", "both"]:
                                    action_values.append(ma_action.get("right_gripper.pos", 0.0))
                                if self.arms in ["left", "both"]:
                                    action_values.append(ma_action.get("left_gripper.pos", 0.0))
                                frame["action"] = np.array(action_values, dtype=np.float32)
                            else:
                                # 마스터 암 상태 없으면 현재 위치 사용
                                frame["action"] = np.array(state_values, dtype=np.float32)
                        else:
                            # 일반 모드: 현재 위치를 action으로
                            frame["action"] = np.array(state_values, dtype=np.float32)

                        # ===== 추가 정보 =====
                        # 속도 벡터
                        velocity_values = [raw_obs.get(f"{name}.vel", 0.0) for name in self.joint_names]
                        frame["observation.velocity"] = np.array(velocity_values, dtype=np.float32)
                        
                        # 토크 벡터
                        effort_values = [raw_obs.get(f"{name}.torque", 0.0) for name in self.joint_names]
                        frame["observation.effort"] = np.array(effort_values, dtype=np.float32)

                        # EEF pose (dynamics 모델 사용 가능시)
                        if self.dyn_robot is not None:
                            eef_values = []
                            if self.arms in ["right", "both"]:
                                eef_values.extend([
                                    raw_obs.get("right_eef.pos_x", 0.0),
                                    raw_obs.get("right_eef.pos_y", 0.0),
                                    raw_obs.get("right_eef.pos_z", 0.0),
                                    raw_obs.get("right_eef.rot_roll", 0.0),
                                    raw_obs.get("right_eef.rot_pitch", 0.0),
                                    raw_obs.get("right_eef.rot_yaw", 0.0),
                                ])
                            if self.arms in ["left", "both"]:
                                eef_values.extend([
                                    raw_obs.get("left_eef.pos_x", 0.0),
                                    raw_obs.get("left_eef.pos_y", 0.0),
                                    raw_obs.get("left_eef.pos_z", 0.0),
                                    raw_obs.get("left_eef.rot_roll", 0.0),
                                    raw_obs.get("left_eef.rot_pitch", 0.0),
                                    raw_obs.get("left_eef.rot_yaw", 0.0),
                                ])
                            frame["observation.eef_pos"] = np.array(eef_values, dtype=np.float32)

                        # 카메라 이미지 (멀티 카메라 지원)
                        if use_cam:
                            if self.rs_pipelines:
                                # 멀티 RealSense 카메라
                                for cam_name in self.rs_pipelines.keys():
                                    if cam_name in raw_obs:
                                        frame[f"observation.images.{cam_name}"] = raw_obs[cam_name]
                            elif "camera" in raw_obs:
                                # 단일 USB 카메라
                                frame["observation.images.camera"] = raw_obs["camera"]

                        # 프레임 추가
                        dataset.add_frame(frame)
                        frame_count += 1

                        # 진행 상황 출력 (매 초)
                        if frame_count % fps == 0:
                            status_parts = []
                            
                            if "right_arm_0" in self.joint_names:
                                r_arm = raw_obs.get("right_arm_0.pos", 0)
                                r_grip = raw_obs.get("right_gripper.pos", 0)
                                status_parts.append(f"R0:{r_arm:.2f} G:{r_grip:.2f}")
                            
                            if "left_arm_0" in self.joint_names:
                                l_arm = raw_obs.get("left_arm_0.pos", 0)
                                l_grip = raw_obs.get("left_gripper.pos", 0)
                                status_parts.append(f"L0:{l_arm:.2f} G:{l_grip:.2f}")
                            
                            joint_info = " | ".join(status_parts)
                            remaining = MAX_EPISODE_DURATION - elapsed
                            print(f"\r  ● REC {elapsed:5.1f}s | 프레임: {frame_count:5d} | {joint_info} | 남은: {remaining:.0f}s  ", end="", flush=True)

                        # FPS 유지
                        elapsed_frame = time.perf_counter() - loop_start
                        sleep_time = frame_interval - elapsed_frame
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                # 에피소드 완료 처리
                if episode_cancelled:
                    if frame_count > 0:  # 현재 녹화 버퍼가 있으면 삭제
                        dataset.clear_episode_buffer()
                    print(f"에피소드 {episode_idx + 1} 취소됨. 다시 녹화합니다.")
                else:
                    dataset.save_episode()
                    episode_frame_counts.append(frame_count)  # 프레임 수 저장
                    total_frames += frame_count
                    print(f"✓ 에피소드 {episode_idx + 1} 저장 완료! ({frame_count} 프레임)")
                    episode_idx += 1

        # 최종 저장
        dataset.finalize()
        self._print_summary(output_name, episode_idx, total_frames, save_root)

        return dataset

    def _print_summary(self, output_name: str, num_episodes: int, total_frames: int, save_root: Path):
        """녹화 완료 요약 출력"""
        print("\n" + "=" * 60)
        print("녹화 완료!")
        print("=" * 60)
        print(f"  저장된 에피소드: {num_episodes}")
        print(f"  총 프레임: {total_frames}")
        print(f"  저장 경로: {save_root / output_name}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RBY1 SDK LeRobot 형식 데이터 로깅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
키보드 조작:
  SPACE  : 녹화 시작/일시정지
  ENTER  : 현재 에피소드 저장 & 다음 에피소드
  R      : 현재 에피소드 취소 & 다시 녹화
  Q      : 종료

예제:
  # 5개 에피소드 녹화 (기본 모드: observation.state = action)
  python record_rby1_standalone.py --address 192.168.30.1:50051 --episodes 5

  # 텔레오퍼레이션 모드 (마스터 암에서 action 기록)
  python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --episodes 5

  # 카메라 + 텔레오프 포함 녹화
  python record_rby1_standalone.py --address 192.168.30.1:50051 --camera 0 --teleop --episodes 3

  # 오른팔만 10개 에피소드
  python record_rby1_standalone.py --address 192.168.30.1:50051 --arms right --episodes 10
  
  # 카메라 웹 스트리밍과 함께 녹화 (http://localhost:8000)
  python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --stream 8000 --episodes 5
        """
    )

    parser.add_argument("--address", type=str, default="192.168.30.1:50051",
                        help="로봇 주소 (기본: 192.168.30.1:50051)")
    parser.add_argument("--model", type=str, default="a", choices=["a", "m", "ub"],
                        help="로봇 모델 (기본: a)")
    parser.add_argument("--arms", type=str, default="right", choices=["right", "left", "both"],
                        help="기록할 팔 선택: right, left, both (기본: right)")
    parser.add_argument("--teleop", action="store_true",
                        help="텔레오퍼레이션 모드: 마스터 암에서 action 기록 (기본: false)")
    parser.add_argument("--camera", type=int, default=None,
                        help="일반 USB 카메라 ID (예: 0, 1)")
    parser.add_argument("--no-realsense", action="store_true",
                        help="RealSense 카메라 비활성화 (기본: RealSense 사용)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="카메라 이름 (쉼표 구분, 예: cam_high,cam_left_wrist,cam_right_wrist)")
    parser.add_argument("--stream", type=int, default=0,
                        help="카메라 웹 스트리밍 포트 (예: 8000, 0이면 비활성화)")
    parser.add_argument("--mode", type=str, default="impedance", choices=["position", "impedance"],
                        help="제어 모드: position(정밀) 또는 impedance(유연, 기본값)")
    parser.add_argument("--no-reset", action="store_true",
                        help="에피소드마다 초기 자세 리셋 비활성화 (기본: 매 에피소드 리셋)")
    parser.add_argument("--wheels", action="store_true",
                        help="[개발중] 휠(wheel) 데이터 기록 활성화")
    parser.add_argument("--fps", type=int, default=30,
                        help="녹화 FPS (기본: 30)")
    parser.add_argument("--episodes", "-e", type=int, default=1,
                        help="녹화할 에피소드 수 (기본: 1)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 데이터셋 이름 (기본: rby1_YYYYMMDD_HHMMSS)")
    parser.add_argument("--task", type=str, default=None,
                        help="태스크 설명 (자연어 instruction)")

    args = parser.parse_args()

    # Task 입력 (인자로 주어지지 않으면 프롬프트)
    if args.task is None:
        print("\n" + "=" * 60)
        print("Task Description 입력")
        print("=" * 60)
        print("예시: 'Pick up the red block and place it on the table'")
        print("      'Open the drawer and grab the object inside'")
        print("=" * 60)
        args.task = input("Task: ").strip()
        if not args.task:
            args.task = "Demonstration recording"
            print(f"(기본값 사용: '{args.task}')")

    # 출력 이름 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"rby1_{timestamp}"

    # use_realsense: 기본값 True, --no-realsense로 비활성화
    use_realsense = not args.no_realsense

    # 카메라 이름 파싱
    camera_names = None
    if args.cameras:
        camera_names = [name.strip() for name in args.cameras.split(",")]

    # 레코더 생성
    recorder = RBY1Recorder(
        address=args.address,
        model=args.model,
        camera_id=args.camera,
        arms=args.arms,
        use_realsense=use_realsense,
        use_teleop=args.teleop,
        camera_names=camera_names,
        stream_port=args.stream,
        control_mode=args.mode,
        reset_pose=not args.no_reset,
        use_wheels=args.wheels,
    )

    try:
        # 연결
        recorder.connect()

        # 에피소드 녹화
        recorder.record_episodes(
            output_name=args.output,
            task=args.task,
            num_episodes=args.episodes,
            fps=args.fps,
            use_camera=args.camera is not None or use_realsense,
        )

    finally:
        recorder.disconnect()


if __name__ == "__main__":
    main()
