#!/usr/bin/env python

"""
RBY1 SDK 텔레오퍼레이션 + LeRobot 형식 데이터 로깅

마스터 암 텔레오퍼레이션과 함께 데이터를 LeRobot 형식으로 기록합니다.
- observation: 팔로워 로봇의 현재 상태 (위치, 속도, 토크, EEF pose)
- action: 마스터 암의 목표 위치 (다음 프레임의 목표)

키보드 조작:
    SPACE : 녹화 시작/중지 토글
    ENTER : 현재 에피소드 저장하고 다음 에피소드로
    Q     : 종료
    R     : 현재 에피소드 취소하고 다시 녹화

마스터 암 조작:
    버튼 누름 : 팔 조작 활성화
    트리거   : 그리퍼 제어

사용 방법:
    # 기본 사용 (텔레오퍼레이션 모드)
    python record_rby1_standalone.py --address 192.168.30.1:50051 --episodes 10

    # 카메라 포함
    python record_rby1_standalone.py --address 192.168.30.1:50051 --camera 0 --episodes 5
    
    # 텔레오퍼레이션 없이 (관측만)
    python record_rby1_standalone.py --address 192.168.30.1:50051 --no-teleop --episodes 5
"""

import argparse
import time
import signal
import sys
import threading
import termios
import tty
import select
from pathlib import Path
from datetime import datetime

import numpy as np

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

# 에피소드당 최대 시간 (초) - 1분
MAX_EPISODE_DURATION = 60

# 마스터 암 루프 주기
MASTER_ARM_LOOP_PERIOD = 1 / 100  # 100Hz

# RBY1-A 조인트 이름 (팔별로 분리)
RIGHT_ARM_JOINTS = [
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
    "right_arm_4", "right_arm_5", "right_arm_6",
]

LEFT_ARM_JOINTS = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
    "left_arm_4", "left_arm_5", "left_arm_6",
]

# 그리퍼 (필요시 추가)
RIGHT_GRIPPER = ["right_gripper"]
LEFT_GRIPPER = ["left_gripper"]

# 모델별 Ready Pose (팔 7개 조인트씩)
READY_POSE = {
    "a": {
        "torso": [0, -0.26, 0, 0.52, 0, -0.26],
        "right_arm": [0.3, 0.3, 0.0, -2.1, 0.0, 0.0, 0.0],
        "left_arm": [-0.3, -0.3, 0.0, 2.1, 0.0, 0.0, 0.0],
    },
}


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

    def get_key(self, timeout: float = 0.01) -> str | None:
        """비차단으로 키 입력 확인"""
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None


class RBY1Recorder:
    """RBY1 SDK를 사용한 LeRobot 형식 데이터 레코더 (마스터 암 텔레오퍼레이션 지원)"""

    def __init__(self, address: str, model: str = "a", camera_id: int | None = None, 
                 arms: str = "both", use_teleop: bool = True):
        self.address = address
        self.model = model
        self.camera_id = camera_id
        self.arms = arms
        self.use_teleop = use_teleop

        self.robot = None
        self.camera = None
        self.master_arm = None
        self.stream = None
        
        # 상태 데이터
        self.latest_state = None
        self.state_lock = threading.Lock()
        self.running = False

        # 마스터 암 상태
        self.master_arm_state = None
        self.master_arm_lock = threading.Lock()
        self.right_target_q = None  # 오른팔 목표 관절 위치
        self.left_target_q = None   # 왼팔 목표 관절 위치
        
        # 마스터 암 버튼/트리거 상태
        self.right_button_active = False
        self.left_button_active = False
        self.right_trigger = 0.0
        self.left_trigger = 0.0

        # 선택한 팔에 따른 조인트 이름 설정
        self.joint_names = self._get_joint_names(arms)

        # EEF pose 관련
        self.dyn_robot = None
        self.dyn_state = None
        self.robot_model = None
        self.prev_eef_pose = {}  # 이전 EEF pose 저장 (delta 계산용)

    def _get_joint_names(self, arms: str) -> list[str]:
        """선택한 팔에 따른 조인트 이름 반환"""
        if arms == "right":
            return RIGHT_ARM_JOINTS.copy()
        elif arms == "left":
            return LEFT_ARM_JOINTS.copy()
        elif arms == "both":
            return RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
        else:
            raise ValueError(f"Invalid arms option: {arms}. Use 'right', 'left', or 'both'")

    def _state_callback(self, robot_state, control_manager_state=None):
        """로봇 상태 업데이트 콜백"""
        with self.state_lock:
            self.latest_state = robot_state

    def connect(self):
        """로봇 및 카메라, 마스터 암 연결"""
        print(f"로봇 연결 중: {self.address}")
        self.robot = rby.create_robot(self.address, self.model)
        self.robot.connect()

        if not self.robot.is_connected():
            raise ConnectionError("로봇 연결 실패")

        print("✓ 로봇 연결됨")

        # 파워 상태 확인 (필요시 파워온)
        if not self.robot.is_power_on(".*"):
            print("파워 온 중...")
            if not self.robot.power_on(".*"):
                raise RuntimeError("파워 온 실패")
            print("✓ 파워 온 완료")

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

        # 마스터 암 연결 (텔레오퍼레이션 모드일 때)
        if self.use_teleop:
            self._connect_master_arm()

        # 카메라 연결
        if self.camera_id is not None:
            try:
                import cv2
                self.camera = cv2.VideoCapture(self.camera_id)
                if not self.camera.isOpened():
                    print(f"⚠ 카메라 {self.camera_id} 열기 실패")
                    self.camera = None
                else:
                    # 카메라 설정
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"✓ 카메라 {self.camera_id} 연결됨")
            except ImportError:
                print("⚠ OpenCV 없음, 카메라 비활성화")
                self.camera = None

    def _connect_master_arm(self):
        """마스터 암 연결 및 초기화"""
        try:
            print("마스터 암 연결 중...")
            self.master_arm = rby.upc.MasterArm(".*")
            
            # 초기 목표 위치 설정 (현재 로봇 위치로)
            with self.state_lock:
                state = self.latest_state
            
            if state is not None:
                positions = np.array(state.position)
                self.right_target_q = positions[self.robot_model.right_arm_idx].copy()
                self.left_target_q = positions[self.robot_model.left_arm_idx].copy()
            else:
                # Ready pose로 초기화
                ready = READY_POSE.get(self.model, READY_POSE["a"])
                self.right_target_q = np.array(ready["right_arm"])
                self.left_target_q = np.array(ready["left_arm"])
            
            print("✓ 마스터 암 연결됨")
            
            # 서보 활성화
            print("서보 활성화 중...")
            if not self.robot.is_servo_on("torso_.*|right_arm_.*|left_arm_.*"):
                if not self.robot.servo_on("torso_.*|right_arm_.*|left_arm_.*"):
                    raise RuntimeError("서보 온 실패")
            print("✓ 서보 활성화됨")
            
            # 컨트롤 매니저 활성화
            if not self.robot.is_control_manager_enabled():
                print("컨트롤 매니저 활성화 중...")
                if not self.robot.enable_control_manager():
                    raise RuntimeError("컨트롤 매니저 활성화 실패")
                print("✓ 컨트롤 매니저 활성화됨")
            
            # 명령 스트림 생성
            self.stream = self.robot.create_command_stream(priority=1)
            
            # Ready pose로 이동
            ready = READY_POSE.get(self.model, READY_POSE["a"])
            ready_cmd = self._build_joint_position_command(
                ready["torso"] + ready["right_arm"] + ready["left_arm"],
                minimum_time=3.0
            )
            self.stream.send_command(ready_cmd)
            print("✓ Ready pose로 이동 중... (3초)")
            time.sleep(3.5)
            
        except Exception as e:
            print(f"⚠ 마스터 암 연결 실패: {e}")
            print("  텔레오퍼레이션 비활성화됨. observation 모드로 동작합니다.")
            self.master_arm = None
            self.use_teleop = False

    def _build_joint_position_command(self, positions: list, minimum_time: float = 1.0):
        """조인트 위치 명령 생성"""
        body_cmd = rby.BodyComponentBasedCommandBuilder()
        
        # Torso
        torso_builder = rby.JointPositionCommandBuilder()
        torso_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(1e6)
        ).set_position(positions[0:6]).set_minimum_time(minimum_time)
        body_cmd.set_torso_command(torso_builder)
        
        # Right arm
        right_builder = rby.JointPositionCommandBuilder()
        right_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(1e6)
        ).set_position(positions[6:13]).set_minimum_time(minimum_time)
        body_cmd.set_right_arm_command(right_builder)
        
        # Left arm
        left_builder = rby.JointPositionCommandBuilder()
        left_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(1e6)
        ).set_position(positions[13:20]).set_minimum_time(minimum_time)
        body_cmd.set_left_arm_command(left_builder)
        
        return rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
        )

    def _master_arm_control_callback(self, state: 'rby.upc.MasterArm.State'):
        """마스터 암 상태 콜백 - 목표 위치 업데이트"""
        with self.master_arm_lock:
            # 버튼 상태
            self.right_button_active = state.button_right.button == 1
            self.left_button_active = state.button_left.button == 1
            self.right_trigger = state.button_right.trigger / 1000.0
            self.left_trigger = state.button_left.trigger / 1000.0
            
            # 마스터 암 관절 위치를 목표로 저장
            if self.right_button_active:
                self.right_target_q = state.q_joint[0:7].copy()
            if self.left_button_active:
                self.left_target_q = state.q_joint[7:14].copy()
        
        # 마스터 암 입력 생성 (피드백 토크)
        ma_input = rby.upc.MasterArm.ControlInput()
        
        # 마스터 암 제한 및 토크 설정
        ma_min_q = np.deg2rad([-360, -90, 0, -60, 90, -80, -360, -360, 30, 0, -60, 90, -80, -360])
        ma_max_q = np.deg2rad([360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360])
        ma_torque_limit = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
        ma_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)
        
        ma_q_limit_barrier = 40
        torque = (
            state.gravity_term
            + ma_q_limit_barrier * (
                np.maximum(ma_min_q - state.q_joint, 0)
                + np.minimum(ma_max_q - state.q_joint, 0)
            )
            + ma_viscous_gain * state.qvel_joint
        )
        torque = np.clip(torque, -ma_torque_limit, ma_torque_limit)
        
        # 오른팔 모드 설정
        if self.right_button_active:
            ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentControlMode)
            ma_input.target_torque[0:7] = torque[0:7] * 0.6
        else:
            ma_input.target_operating_mode[0:7].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            ma_input.target_torque[0:7] = ma_torque_limit[0:7]
            with self.master_arm_lock:
                ma_input.target_position[0:7] = self.right_target_q
        
        # 왼팔 모드 설정
        if self.left_button_active:
            ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentControlMode)
            ma_input.target_torque[7:14] = torque[7:14] * 0.6
        else:
            ma_input.target_operating_mode[7:14].fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            ma_input.target_torque[7:14] = ma_torque_limit[7:14]
            with self.master_arm_lock:
                ma_input.target_position[7:14] = self.left_target_q
        
        # 로봇에 명령 전송 (버튼이 눌렸을 때만)
        if self.stream is not None and (self.right_button_active or self.left_button_active):
            self._send_robot_command()
        
        return ma_input

    def _send_robot_command(self):
        """로봇에 목표 위치 명령 전송"""
        with self.master_arm_lock:
            right_q = self.right_target_q.copy() if self.right_target_q is not None else None
            left_q = self.left_target_q.copy() if self.left_target_q is not None else None
        
        if right_q is None or left_q is None:
            return
            
        # 로봇 관절 제한 가져오기
        robot_min_q = np.array(self.robot_model.robot_min_q)
        robot_max_q = np.array(self.robot_model.robot_max_q)
        
        rc = rby.BodyComponentBasedCommandBuilder()
        
        # 오른팔 명령
        if self.right_button_active:
            right_builder = rby.JointPositionCommandBuilder()
            right_builder.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(1e6)
            ).set_position(
                np.clip(right_q, robot_min_q[self.robot_model.right_arm_idx], 
                       robot_max_q[self.robot_model.right_arm_idx])
            ).set_minimum_time(MASTER_ARM_LOOP_PERIOD * 1.01)
            rc.set_right_arm_command(right_builder)
        
        # 왼팔 명령
        if self.left_button_active:
            left_builder = rby.JointPositionCommandBuilder()
            left_builder.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(1e6)
            ).set_position(
                np.clip(left_q, robot_min_q[self.robot_model.left_arm_idx],
                       robot_max_q[self.robot_model.left_arm_idx])
            ).set_minimum_time(MASTER_ARM_LOOP_PERIOD * 1.01)
            rc.set_left_arm_command(left_builder)
        
        self.stream.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(rc)
            )
        )

    def start_teleop(self):
        """텔레오퍼레이션 시작"""
        if self.master_arm is not None:
            self.master_arm.start_control(self._master_arm_control_callback)
            print("✓ 텔레오퍼레이션 시작됨")
            print("  버튼을 누르면서 마스터 암을 움직여 로봇을 조작하세요.")

    def stop_teleop(self):
        """텔레오퍼레이션 중지"""
        if self.master_arm is not None:
            self.master_arm.stop_control()

    def disconnect(self):
        """연결 해제"""
        if self.master_arm:
            self.stop_teleop()
            print("✓ 마스터 암 연결 해제")
        
        if self.robot:
            self.robot.stop_state_update()
            if self.stream:
                self.robot.cancel_control()
            print("✓ 로봇 연결 해제")

        if self.camera:
            self.camera.release()
            print("✓ 카메라 연결 해제")

    def get_observation(self) -> dict:
        """현재 관측 데이터 수집 (팔로워 로봇 상태)"""
        obs = {}

        # 로봇 상태 (팔로워)
        with self.state_lock:
            state = self.latest_state

        if state is not None:
            positions = np.array(state.position)
            velocities = np.array(state.velocity)
            torques = np.array(state.torque)

            # 선택한 팔의 관절 인덱스 가져오기
            if self.arms == "right":
                joint_indices = self.robot_model.right_arm_idx if self.robot_model else list(range(6, 13))
            elif self.arms == "left":
                joint_indices = self.robot_model.left_arm_idx if self.robot_model else list(range(13, 20))
            else:  # both
                right_idx = self.robot_model.right_arm_idx if self.robot_model else list(range(6, 13))
                left_idx = self.robot_model.left_arm_idx if self.robot_model else list(range(13, 20))
                joint_indices = list(right_idx) + list(left_idx)

            for i, name in enumerate(self.joint_names):
                if i < len(joint_indices) and joint_indices[i] < len(positions):
                    idx = joint_indices[i]
                    obs[f"{name}.pos"] = float(positions[idx])
                    obs[f"{name}.vel"] = float(velocities[idx])
                    obs[f"{name}.torque"] = float(torques[idx])

            # EEF pose 계산
            if self.dyn_robot is not None:
                try:
                    self._compute_eef_pose(positions, obs)
                except Exception as e:
                    pass  # EEF 계산 실패시 무시

        # 카메라 이미지
        if self.camera is not None:
            import cv2
            ret, frame = self.camera.read()
            if ret:
                # BGR -> RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs["camera"] = frame_rgb

        return obs

    def get_action(self) -> dict:
        """마스터 암 목표 위치 (action) 가져오기"""
        action = {}
        
        with self.master_arm_lock:
            right_q = self.right_target_q.copy() if self.right_target_q is not None else None
            left_q = self.left_target_q.copy() if self.left_target_q is not None else None
            right_trigger = self.right_trigger
            left_trigger = self.left_trigger
        
        # 선택한 팔에 따라 action 구성
        if self.arms in ["right", "both"] and right_q is not None:
            for i, name in enumerate(RIGHT_ARM_JOINTS):
                if i < len(right_q):
                    action[f"{name}.pos"] = float(right_q[i])
            action["right_gripper"] = float(right_trigger)
        
        if self.arms in ["left", "both"] and left_q is not None:
            for i, name in enumerate(LEFT_ARM_JOINTS):
                if i < len(left_q):
                    action[f"{name}.pos"] = float(left_q[i])
            action["left_gripper"] = float(left_trigger)
        
        return action

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

    def build_features(self, use_camera: bool = False, camera_shape: tuple = (480, 640, 3)) -> dict:
        """데이터셋 feature 정의 생성"""
        features = {}

        # 조인트 상태
        for name in self.joint_names:
            features[f"observation.{name}.pos"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }
            features[f"observation.{name}.vel"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }
            features[f"observation.{name}.torque"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }
            # 액션은 위치만
            features[f"action.{name}.pos"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }

        # EEF pose (dynamics 모델 사용 가능시)
        if self.dyn_robot is not None:
            eef_arms = []
            if self.arms in ["right", "both"]:
                eef_arms.append("right")
            if self.arms in ["left", "both"]:
                eef_arms.append("left")
            
            for arm in eef_arms:
                # Absolute pose
                for suffix in ["pos_x", "pos_y", "pos_z", "rot_roll", "rot_pitch", "rot_yaw"]:
                    features[f"observation.{arm}_eef.{suffix}"] = {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": None,
                    }
                # Delta pose
                for suffix in ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw"]:
                    features[f"observation.{arm}_eef.{suffix}"] = {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": None,
                    }
                    # Action으로도 delta pose 기록
                    features[f"action.{arm}_eef.{suffix}"] = {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": None,
                    }

        # 카메라
        if use_camera:
            features["observation.camera"] = {
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
        print(f"  텔레오퍼레이션: {'활성화' if self.use_teleop and self.master_arm else '비활성화 (관측만)'}")
        print(f"  카메라: {'활성화' if use_camera and self.camera else '비활성화'}")
        print("=" * 60)
        print("\n키보드 조작:")
        print("  [SPACE] 녹화 시작/일시정지")
        print("  [ENTER] 에피소드 저장 & 다음으로")
        print("  [R]     에피소드 취소 & 다시 녹화")
        print("  [Q]     종료")
        if self.use_teleop and self.master_arm:
            print("\n마스터 암 조작:")
            print("  버튼 누름 : 팔 조작 활성화")
            print("  트리거   : 그리퍼 제어")
        print("=" * 60)

        # Feature 정의
        use_cam = use_camera and self.camera is not None
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

        # 텔레오퍼레이션 시작
        if self.use_teleop and self.master_arm:
            self.start_teleop()

        frame_interval = 1.0 / fps
        episode_idx = 0
        total_frames = 0

        with KeyboardController() as keyboard:
            while episode_idx < num_episodes:
                print(f"\n{'='*60}")
                print(f"에피소드 {episode_idx + 1}/{num_episodes}")
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

                        elif key.lower() == 'r':  # R - 에피소드 취소
                            if frame_count > 0:
                                episode_cancelled = True
                                episode_done = True
                                print("\n✗ 에피소드 취소됨")
                            else:
                                print("\n취소할 녹화가 없습니다.")

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

                    # 녹화 중일 때 프레임 수집
                    if recording:
                        loop_start = time.perf_counter()
                        elapsed = time.time() - episode_start_time

                        # 최대 시간 체크
                        if elapsed >= MAX_EPISODE_DURATION:
                            print(f"\n⏱ 최대 시간({MAX_EPISODE_DURATION}초) 도달! 에피소드 자동 저장...")
                            episode_done = True
                            continue

                        # 관측 수집 (팔로워 로봇 현재 상태)
                        raw_obs = self.get_observation()
                        
                        # 액션 수집 (마스터 암 목표 위치 또는 현재 위치)
                        if self.use_teleop and self.master_arm:
                            raw_action = self.get_action()
                        else:
                            # 텔레오퍼레이션이 없으면 현재 위치를 action으로 사용
                            raw_action = {f"{name}.pos": raw_obs.get(f"{name}.pos", 0.0) for name in self.joint_names}

                        # 프레임 구성
                        frame = {"task": task}

                        for name in self.joint_names:
                            # Observation: 팔로워 로봇의 현재 상태
                            frame[f"observation.{name}.pos"] = np.array([raw_obs.get(f"{name}.pos", 0.0)], dtype=np.float32)
                            frame[f"observation.{name}.vel"] = np.array([raw_obs.get(f"{name}.vel", 0.0)], dtype=np.float32)
                            frame[f"observation.{name}.torque"] = np.array([raw_obs.get(f"{name}.torque", 0.0)], dtype=np.float32)
                            
                            # Action: 마스터 암의 목표 위치 (텔레오퍼레이션 사용시)
                            frame[f"action.{name}.pos"] = np.array([raw_action.get(f"{name}.pos", 0.0)], dtype=np.float32)

                        # EEF pose 추가
                        if self.dyn_robot is not None:
                            eef_arms = []
                            if self.arms in ["right", "both"]:
                                eef_arms.append("right")
                            if self.arms in ["left", "both"]:
                                eef_arms.append("left")
                            
                            for arm in eef_arms:
                                # Absolute pose
                                for suffix in ["pos_x", "pos_y", "pos_z", "rot_roll", "rot_pitch", "rot_yaw"]:
                                    key = f"{arm}_eef.{suffix}"
                                    frame[f"observation.{key}"] = np.array([raw_obs.get(key, 0.0)], dtype=np.float32)
                                # Delta pose
                                for suffix in ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw"]:
                                    key = f"{arm}_eef.{suffix}"
                                    frame[f"observation.{key}"] = np.array([raw_obs.get(key, 0.0)], dtype=np.float32)
                                    frame[f"action.{key}"] = np.array([raw_obs.get(key, 0.0)], dtype=np.float32)

                        if use_cam and "camera" in raw_obs:
                            frame["observation.camera"] = raw_obs["camera"]

                        # 프레임 추가
                        dataset.add_frame(frame)
                        frame_count += 1

                        # 진행 상황 출력 (매 초)
                        if frame_count % fps == 0:
                            if "right_arm_0" in self.joint_names:
                                r_arm = raw_obs.get("right_arm_0.pos", 0)
                                r_target = raw_action.get("right_arm_0.pos", 0)
                                r_str = f"R: {r_arm:.2f}→{r_target:.2f}"
                            else:
                                r_str = ""
                            
                            if "left_arm_0" in self.joint_names:
                                l_arm = raw_obs.get("left_arm_0.pos", 0)
                                l_target = raw_action.get("left_arm_0.pos", 0)
                                l_str = f"L: {l_arm:.2f}→{l_target:.2f}"
                            else:
                                l_str = ""
                            
                            # 마스터 암 버튼 상태
                            if self.use_teleop and self.master_arm:
                                btn_r = "●" if self.right_button_active else "○"
                                btn_l = "●" if self.left_button_active else "○"
                                btn_str = f"[{btn_r}R {btn_l}L]"
                            else:
                                btn_str = ""
                            
                            joint_info = " | ".join(filter(None, [r_str, l_str, btn_str]))
                            remaining = MAX_EPISODE_DURATION - elapsed
                            print(f"\r  ● REC {elapsed:5.1f}s | 프레임: {frame_count:5d} | {joint_info} | 남은: {remaining:.0f}s  ", end="", flush=True)

                        # FPS 유지
                        elapsed_frame = time.perf_counter() - loop_start
                        sleep_time = frame_interval - elapsed_frame
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                # 에피소드 완료 처리
                if episode_cancelled:
                    dataset.clear_episode_buffer()
                    print(f"에피소드 {episode_idx + 1} 취소됨. 다시 녹화합니다.")
                else:
                    dataset.save_episode()
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
        description="RBY1 SDK 텔레오퍼레이션 + LeRobot 형식 데이터 로깅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
키보드 조작:
  SPACE  : 녹화 시작/일시정지
  ENTER  : 현재 에피소드 저장 & 다음 에피소드
  R      : 현재 에피소드 취소 & 다시 녹화
  Q      : 종료

마스터 암 조작:
  버튼 누름 : 팔 조작 활성화 (버튼 누른 상태에서만 로봇이 움직임)
  트리거   : 그리퍼 제어

예제:
  # 텔레오퍼레이션으로 5개 에피소드 녹화
  python record_rby1_standalone.py --address 192.168.30.1:50051 --episodes 5

  # 카메라 포함 녹화
  python record_rby1_standalone.py --address 192.168.30.1:50051 --camera 0 --episodes 3

  # 오른팔만 10개 에피소드
  python record_rby1_standalone.py --address 192.168.30.1:50051 --arms right --episodes 10
  
  # 텔레오퍼레이션 없이 관측만 기록 (테스트용)
  python record_rby1_standalone.py --address 192.168.30.1:50051 --no-teleop --episodes 3
        """
    )

    parser.add_argument("--address", type=str, default="192.168.30.1:50051",
                        help="로봇 주소 (기본: 192.168.30.1:50051)")
    parser.add_argument("--model", type=str, default="a", choices=["a", "m", "ub"],
                        help="로봇 모델 (기본: a)")
    parser.add_argument("--arms", type=str, default="both", choices=["right", "left", "both"],
                        help="기록할 팔 선택: right, left, both (기본: both)")
    parser.add_argument("--camera", type=int, default=None,
                        help="카메라 ID (예: 0, 1). 지정하지 않으면 카메라 비활성화")
    parser.add_argument("--fps", type=int, default=30,
                        help="녹화 FPS (기본: 30)")
    parser.add_argument("--episodes", "-e", type=int, default=1,
                        help="녹화할 에피소드 수 (기본: 1)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 데이터셋 이름 (기본: rby1_YYYYMMDD_HHMMSS)")
    parser.add_argument("--task", type=str, default="RBY1 teleoperation",
                        help="태스크 설명")
    parser.add_argument("--no-teleop", action="store_true",
                        help="텔레오퍼레이션 비활성화 (관측만 기록)")

    args = parser.parse_args()

    # 출력 이름 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"rby1_{timestamp}"

    # 레코더 생성
    recorder = RBY1Recorder(
        address=args.address,
        model=args.model,
        camera_id=args.camera,
        arms=args.arms,
        use_teleop=not args.no_teleop,  # --no-teleop 플래그 적용
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
            use_camera=args.camera is not None,
        )

    finally:
        recorder.disconnect()


if __name__ == "__main__":
    main()
