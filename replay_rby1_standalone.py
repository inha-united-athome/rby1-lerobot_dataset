#!/usr/bin/env python3
"""
RBY1 LeRobot 데이터셋 안전 재생 스크립트

저장된 데이터셋을 로봇에 안전하게 재생합니다.

=== 안전 기능 ===
1. 속도/가속도 제한 (사용자 설정 가능)
2. 충돌 감지 시 자동 정지
3. 스페이스바로 즉시 일시정지
4. ESC로 긴급 정지
5. 첫 프레임 이동 시 5초 대기 (안전 이동)
6. 토크 제한이 있는 Joint Impedance Control 옵션
7. 드라이런 모드 (로봇 없이 시뮬레이션)

=== 사용법 ===
# 드라이런 (로봇 없이 테스트)
python replay_rby1_safe.py -d dataset_name --dry-run

# 실제 재생 (기본 0.5x 속도)
python replay_rby1_safe.py -d dataset_name --address 192.168.30.1:50051

# 느린 속도로 재생 (0.25x)
python replay_rby1_safe.py -d dataset_name --address 192.168.30.1:50051 --speed 0.25

# 특정 프레임 범위만 재생
python replay_rby1_safe.py -d dataset_name --address 192.168.30.1:50051 --frames 0-100

# Impedance Control 모드 (부드러운 제어)
python replay_rby1_safe.py -d dataset_name --address 192.168.30.1:50051 --impedance
"""

import argparse
import sys
import time
import signal
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

# LeRobot 데이터셋
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("❌ lerobot이 설치되지 않았습니다.")
    sys.exit(1)

# RBY1 SDK
try:
    import rby1_sdk as rby
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("⚠️  rby1_sdk를 찾을 수 없습니다. --dry-run 모드만 가능합니다.")


# 기본 설정
DEFAULT_DATASETS_DIR = Path.home() / "vla_ws" / "datasets"


@dataclass
class SafetySettings:
    """안전 설정"""
    # 속도/가속도 제한 (rad/s, rad/s²)
    max_velocity: float = 1.0        # 최대 관절 속도
    max_acceleration: float = 2.0    # 최대 관절 가속도
    
    # 재생 속도 (1.0 = 원래 속도)
    playback_speed: float = 0.5      # 기본 0.5x 속도
    
    # 첫 프레임 이동 시간 (초)
    initial_move_time: float = 5.0
    
    # 충돌 감지 임계값 (m)
    collision_threshold: float = 0.02
    
    # Impedance Control 설정
    stiffness: float = 100.0         # Nm/rad
    damping_ratio: float = 1.0
    torque_limit: float = 10.0       # Nm
    
    # 제어 주기 (초)
    control_dt: float = 0.01         # 100Hz


class SafeRobotController:
    """안전한 로봇 제어기"""
    
    def __init__(self, address: str, model: str = "a", settings: SafetySettings = None):
        self.address = address
        self.model_name = model
        self.settings = settings or SafetySettings()
        self.robot = None
        self.robot_model = None
        self.stream = None
        
        # 상태 플래그
        self.is_connected = False
        self.is_paused = False
        self.is_emergency_stop = False
        self.collision_detected = False
        self.current_position = None
        
        # 스레드 동기화
        self.lock = threading.Lock()
        
    def connect(self) -> bool:
        """로봇 연결 및 초기화"""
        if not HAS_SDK:
            print("❌ rby1_sdk가 없습니다.")
            return False
            
        print(f"🔌 로봇 연결 중: {self.address}")
        
        try:
            self.robot = rby.create_robot(self.address, self.model_name)
            if not self.robot.connect():
                print("❌ 연결 실패")
                return False
                
            print("✅ 연결 성공")
            
            # 전원 상태 확인
            if not self.robot.is_power_on(".*"):
                print("⚡ 전원 켜는 중...")
                if not self.robot.power_on(".*"):
                    print("❌ 전원 켜기 실패")
                    return False
                    
            # 서보 상태 확인
            if not self.robot.is_servo_on(".*"):
                print("🔧 서보 켜는 중...")
                if not self.robot.servo_on(".*"):
                    print("❌ 서보 켜기 실패")
                    return False
            
            # 결함 확인 및 리셋
            cm_state = self.robot.get_control_manager_state()
            if cm_state.state in [
                rby.ControlManagerState.State.MajorFault,
                rby.ControlManagerState.State.MinorFault,
            ]:
                print("⚠️  결함 감지됨, 리셋 시도...")
                if not self.robot.reset_fault_control_manager():
                    print("❌ 결함 리셋 실패")
                    return False
            
            # 제어 매니저 활성화
            if not self.robot.enable_control_manager():
                print("❌ 제어 매니저 활성화 실패")
                return False
            
            self.robot_model = self.robot.model()
            self.is_connected = True
            
            # 현재 위치 읽기
            state = self.robot.get_state()
            self.current_position = np.array(state.position)
            
            # 명령 스트림 생성
            self.stream = self.robot.create_command_stream(10)
            
            # 충돌 감지 콜백 시작
            self._start_collision_monitor()
            
            print("✅ 초기화 완료")
            print(f"   모델: {self.robot_model.model_name}")
            print(f"   DoF: {self.robot_model.robot_dof}")
            
            return True
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            return False
    
    def _start_collision_monitor(self):
        """충돌 감지 모니터링 시작"""
        def callback(robot_state):
            with self.lock:
                self.current_position = np.array(robot_state.position)
                
                if robot_state.collisions:
                    collision = robot_state.collisions[0]
                    if collision.distance < self.settings.collision_threshold:
                        self.collision_detected = True
                        print(f"\n⚠️  충돌 감지! 거리: {collision.distance:.4f}m")
        
        self.robot.start_state_update(callback, rate=50)  # 50Hz
    
    def emergency_stop(self):
        """긴급 정지"""
        with self.lock:
            self.is_emergency_stop = True
        
        if self.robot:
            print("\n🛑 긴급 정지!")
            self.robot.cancel_control()
    
    def pause(self):
        """일시정지 토글"""
        with self.lock:
            self.is_paused = not self.is_paused
            status = "일시정지" if self.is_paused else "재개"
            print(f"\n⏸️  {status}")
    
    def move_to_position(self, target_position: np.ndarray, minimum_time: float = 5.0,
                         use_impedance: bool = False) -> bool:
        """목표 위치로 이동 (안전하게)"""
        if self.is_emergency_stop:
            return False
            
        # 일시정지 대기
        while self.is_paused and not self.is_emergency_stop:
            time.sleep(0.1)
        
        if self.is_emergency_stop:
            return False
        
        # 충돌 체크
        if self.collision_detected:
            print("❌ 충돌 감지로 인해 이동 취소")
            return False
        
        try:
            # Body 부분만 추출 (wheel 2개 제외, head 2개 제외)
            # Model A: [wheel(2), torso(6), right_arm(7), left_arm(7), head(2)] = 24 DoF
            # Body = torso + right_arm + left_arm = 20 DoF
            body_start = 2  # wheel 다음
            body_end = -2   # head 전까지
            target_body = target_position[body_start:body_end] if len(target_position) > 20 else target_position
            
            if use_impedance:
                # Impedance Control 사용 (부드러운 제어)
                rc = rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder().set_body_command(
                        rby.BodyComponentBasedCommandBuilder()
                        .set_torso_command(
                            rby.JointImpedanceControlCommandBuilder()
                            .set_command_header(
                                rby.CommandHeaderBuilder().set_control_hold_time(minimum_time * 2)
                            )
                            .set_position(target_body[:6].tolist())
                            .set_minimum_time(minimum_time)
                            .set_stiffness([self.settings.stiffness] * 6)
                            .set_damping_ratio(self.settings.damping_ratio)
                            .set_torque_limit([self.settings.torque_limit] * 6)
                        )
                        .set_right_arm_command(
                            rby.JointImpedanceControlCommandBuilder()
                            .set_command_header(
                                rby.CommandHeaderBuilder().set_control_hold_time(minimum_time * 2)
                            )
                            .set_position(target_body[6:13].tolist())
                            .set_minimum_time(minimum_time)
                            .set_stiffness([self.settings.stiffness] * 7)
                            .set_damping_ratio(self.settings.damping_ratio)
                            .set_torque_limit([self.settings.torque_limit] * 7)
                        )
                        .set_left_arm_command(
                            rby.JointImpedanceControlCommandBuilder()
                            .set_command_header(
                                rby.CommandHeaderBuilder().set_control_hold_time(minimum_time * 2)
                            )
                            .set_position(target_body[13:20].tolist())
                            .set_minimum_time(minimum_time)
                            .set_stiffness([self.settings.stiffness] * 7)
                            .set_damping_ratio(self.settings.damping_ratio)
                            .set_torque_limit([self.settings.torque_limit] * 7)
                        )
                    )
                )
            else:
                # 일반 위치 제어
                rc = rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder().set_body_command(
                        rby.JointPositionCommandBuilder()
                        .set_command_header(
                            rby.CommandHeaderBuilder().set_control_hold_time(minimum_time * 2)
                        )
                        .set_minimum_time(minimum_time)
                        .set_position(target_body.tolist())
                        .set_velocity_limit([self.settings.max_velocity] * len(target_body))
                        .set_acceleration_limit([self.settings.max_acceleration] * len(target_body))
                    )
                )
            
            rv = self.stream.send_command(rc)
            return True
            
        except Exception as e:
            print(f"❌ 이동 실패: {e}")
            return False
    
    def disconnect(self):
        """연결 해제"""
        if self.robot:
            print("\n🔌 로봇 연결 해제 중...")
            try:
                self.robot.stop_state_update()
                self.robot.disable_control_manager()
            except:
                pass
            self.is_connected = False
            print("✅ 연결 해제 완료")


class DatasetPlayer:
    """데이터셋 재생기"""
    
    def __init__(self, dataset_path: Path, settings: SafetySettings = None):
        self.dataset_path = dataset_path
        self.settings = settings or SafetySettings()
        self.ds = None
        self.current_frame = 0
        self.total_frames = 0
        
    def load(self) -> bool:
        """데이터셋 로드"""
        try:
            print(f"📂 데이터셋 로드 중: {self.dataset_path.name}")
            self.ds = LeRobotDataset(
                repo_id=f"local/{self.dataset_path.name}",
                root=self.dataset_path,
            )
            self.total_frames = len(self.ds)
            print(f"✅ 로드 완료: {self.total_frames} 프레임, {self.ds.num_episodes} 에피소드")
            return True
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False
    
    def get_frame_positions(self, frame_idx: int) -> Optional[np.ndarray]:
        """프레임에서 관절 위치 추출"""
        if frame_idx >= self.total_frames:
            return None
            
        frame = self.ds[frame_idx]
        
        # action 키 찾기 (보통 'action' 또는 'action.state' 등)
        action_key = None
        for k in frame.keys():
            if 'action' in k.lower() and not 'camera' in k.lower():
                action_key = k
                break
        
        # 또는 observation.state에서 위치 추출
        if action_key is None:
            for k in frame.keys():
                if 'observation.state' in k.lower():
                    action_key = k
                    break
        
        if action_key is None:
            # 개별 관절 위치 찾기
            pos_keys = sorted([k for k in frame.keys() if k.endswith('.pos')])
            if pos_keys:
                positions = []
                for k in pos_keys:
                    v = frame[k]
                    val = v.numpy().item() if hasattr(v, 'numpy') else float(v)
                    positions.append(val)
                return np.array(positions)
            return None
        
        # Tensor to numpy
        v = frame[action_key]
        if hasattr(v, 'numpy'):
            return v.numpy()
        return np.array(v)
    
    def get_trajectory(self, start: int = 0, end: int = None) -> List[np.ndarray]:
        """프레임 범위의 궤적 추출"""
        if end is None:
            end = self.total_frames
        end = min(end, self.total_frames)
        
        trajectory = []
        for i in range(start, end):
            pos = self.get_frame_positions(i)
            if pos is not None:
                trajectory.append(pos)
        return trajectory


def setup_keyboard_handler(controller: Optional[SafeRobotController]):
    """키보드 입력 핸들러 설정"""
    import termios
    import tty
    import select
    
    def handler():
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == ' ':  # 스페이스바
                        if controller:
                            controller.pause()
                    elif key == '\x1b':  # ESC
                        if controller:
                            controller.emergency_stop()
                        break
                    elif key == 'q':  # Q
                        if controller:
                            controller.emergency_stop()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    thread = threading.Thread(target=handler, daemon=True)
    thread.start()
    return thread


def dry_run_replay(player: DatasetPlayer, start: int, end: int, settings: SafetySettings):
    """드라이런 (로봇 없이 시뮬레이션)"""
    print("\n" + "=" * 60)
    print("🎬 드라이런 모드 (로봇 연결 없이 시뮬레이션)")
    print("=" * 60)
    
    trajectory = player.get_trajectory(start, end)
    if not trajectory:
        print("❌ 궤적을 추출할 수 없습니다.")
        return
    
    print(f"📊 궤적 정보:")
    print(f"   프레임: {start} ~ {end-1} ({len(trajectory)} 프레임)")
    print(f"   관절 수: {len(trajectory[0])}")
    print(f"   재생 속도: {settings.playback_speed}x")
    print(f"   예상 재생 시간: {len(trajectory) / player.ds.fps / settings.playback_speed:.1f}초")
    
    print("\n처음 5개 프레임 위치:")
    for i, pos in enumerate(trajectory[:5]):
        print(f"  [{i}] {np.rad2deg(pos[:7])[:4]}... (deg)")
    
    print("\n마지막 5개 프레임 위치:")
    for i, pos in enumerate(trajectory[-5:], start=len(trajectory)-5):
        print(f"  [{i}] {np.rad2deg(pos[:7])[:4]}... (deg)")
    
    # 관절 범위 분석
    trajectory_arr = np.array(trajectory)
    print("\n📈 관절 범위 분석:")
    print(f"   최소: {np.rad2deg(trajectory_arr.min(axis=0)[:7])[:4]}... (deg)")
    print(f"   최대: {np.rad2deg(trajectory_arr.max(axis=0)[:7])[:4]}... (deg)")
    print(f"   범위: {np.rad2deg(trajectory_arr.max(axis=0) - trajectory_arr.min(axis=0))[:7][:4]}... (deg)")
    
    # 속도 분석
    if len(trajectory) > 1:
        dt = 1.0 / player.ds.fps
        velocities = np.diff(trajectory_arr, axis=0) / dt
        max_vel = np.abs(velocities).max(axis=0)
        print(f"\n⚡ 최대 관절 속도:")
        print(f"   {np.rad2deg(max_vel[:7])[:4]}... (deg/s)")
        print(f"   설정된 제한: {np.rad2deg(settings.max_velocity):.1f} deg/s")
        
        if max_vel.max() > settings.max_velocity:
            print(f"   ⚠️  일부 속도가 제한을 초과합니다! (실제 재생 시 제한됨)")
    
    print("\n✅ 드라이런 완료")


def robot_replay(controller: SafeRobotController, player: DatasetPlayer, 
                 start: int, end: int, settings: SafetySettings, use_impedance: bool):
    """실제 로봇 재생"""
    print("\n" + "=" * 60)
    print("🤖 로봇 재생 모드")
    print("=" * 60)
    print("조작키:")
    print("  SPACE : 일시정지/재개")
    print("  ESC/Q : 긴급 정지")
    print("=" * 60)
    
    trajectory = player.get_trajectory(start, end)
    if not trajectory:
        print("❌ 궤적을 추출할 수 없습니다.")
        return
    
    print(f"\n📊 재생 정보:")
    print(f"   프레임: {start} ~ {end-1} ({len(trajectory)} 프레임)")
    print(f"   재생 속도: {settings.playback_speed}x")
    print(f"   Impedance Control: {'ON' if use_impedance else 'OFF'}")
    
    # 확인
    input("\n⚠️  로봇이 움직입니다. 준비되면 Enter를 누르세요...")
    
    # 키보드 핸들러 시작
    setup_keyboard_handler(controller)
    
    # 첫 프레임으로 이동 (천천히)
    print(f"\n🚀 첫 프레임으로 이동 중 ({settings.initial_move_time}초)...")
    first_pos = trajectory[0]
    if not controller.move_to_position(first_pos, settings.initial_move_time, use_impedance):
        print("❌ 첫 프레임 이동 실패")
        return
    
    time.sleep(settings.initial_move_time)
    print("✅ 첫 프레임 도착")
    
    # 재생 시작
    print("\n▶️  재생 시작!")
    dt = 1.0 / player.ds.fps / settings.playback_speed
    
    for i, target_pos in enumerate(trajectory[1:], start=1):
        if controller.is_emergency_stop:
            print("\n🛑 긴급 정지로 재생 중단")
            break
            
        if controller.collision_detected:
            print("\n⚠️  충돌 감지로 재생 중단")
            break
        
        # 일시정지 대기
        while controller.is_paused and not controller.is_emergency_stop:
            time.sleep(0.1)
        
        if controller.is_emergency_stop:
            break
        
        # 진행 상황 출력
        progress = (i + 1) / len(trajectory) * 100
        print(f"\r  [{i+1}/{len(trajectory)}] {progress:.1f}% ", end="", flush=True)
        
        # 목표 위치로 이동
        move_time = dt * 0.9  # 약간의 여유
        if not controller.move_to_position(target_pos, move_time, use_impedance):
            print(f"\n❌ 프레임 {i} 이동 실패")
            break
        
        time.sleep(dt * 0.95)
    
    print("\n\n✅ 재생 완료")


def main():
    parser = argparse.ArgumentParser(
        description="RBY1 LeRobot 데이터셋 안전 재생",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="재생할 데이터셋 이름")
    parser.add_argument("--datasets-dir", type=str, default=None,
                        help=f"데이터셋 폴더 (기본: {DEFAULT_DATASETS_DIR})")
    parser.add_argument("--frames", "-f", type=str, default=None,
                        help="재생할 프레임 범위 (예: 0-100)")
    
    # 로봇 연결
    parser.add_argument("--address", "-a", type=str, default="192.168.30.1:50051",
                        help="로봇 주소 (기본: 192.168.30.1:50051)")
    parser.add_argument("--model", "-m", type=str, default="a",
                        help="로봇 모델 (a/m/ub, 기본: a)")
    parser.add_argument("--dry-run", action="store_true",
                        help="드라이런 모드 (로봇 없이 시뮬레이션)")
    
    # 안전 설정
    parser.add_argument("--speed", type=float, default=0.5,
                        help="재생 속도 (0.1~1.0, 기본: 0.5)")
    parser.add_argument("--max-vel", type=float, default=1.0,
                        help="최대 관절 속도 rad/s (기본: 1.0)")
    parser.add_argument("--max-acc", type=float, default=2.0,
                        help="최대 관절 가속도 rad/s² (기본: 2.0)")
    parser.add_argument("--impedance", action="store_true",
                        help="Impedance Control 사용 (부드러운 제어)")
    parser.add_argument("--stiffness", type=float, default=100.0,
                        help="Impedance 강성 Nm/rad (기본: 100)")
    parser.add_argument("--torque-limit", type=float, default=10.0,
                        help="토크 제한 Nm (기본: 10)")
    
    args = parser.parse_args()
    
    # 안전 설정
    settings = SafetySettings(
        max_velocity=args.max_vel,
        max_acceleration=args.max_acc,
        playback_speed=min(1.0, max(0.1, args.speed)),
        stiffness=args.stiffness,
        torque_limit=args.torque_limit,
    )
    
    # 데이터셋 경로
    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else DEFAULT_DATASETS_DIR
    dataset_path = datasets_dir / args.dataset
    
    if not dataset_path.exists():
        print(f"❌ 데이터셋을 찾을 수 없습니다: {dataset_path}")
        sys.exit(1)
    
    # 데이터셋 로드
    player = DatasetPlayer(dataset_path, settings)
    if not player.load():
        sys.exit(1)
    
    # 프레임 범위
    start, end = 0, player.total_frames
    if args.frames:
        if "-" in args.frames:
            start, end = map(int, args.frames.split("-"))
        else:
            start = int(args.frames)
            end = start + 1
    end = min(end, player.total_frames)
    
    # 드라이런 모드
    if args.dry_run or not HAS_SDK:
        dry_run_replay(player, start, end, settings)
        return
    
    # 실제 재생
    controller = SafeRobotController(args.address, args.model, settings)
    
    # 시그널 핸들러 설정
    def signal_handler(sig, frame):
        print("\n⚠️  인터럽트 감지")
        controller.emergency_stop()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if not controller.connect():
            sys.exit(1)
        
        robot_replay(controller, player, start, end, settings, args.impedance)
        
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
