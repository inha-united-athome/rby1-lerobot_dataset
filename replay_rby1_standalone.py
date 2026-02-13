#!/usr/bin/env python

"""
RBY1 LeRobot 데이터셋 읽기 및 재생 스크립트

저장된 데이터셋을 읽고 확인하거나 로봇에 재생합니다.

사용 방법:
    # 데이터셋 목록 보기
    python replay_rby1_standalone.py --list

    # 특정 데이터셋 정보 확인
    python replay_rby1_standalone.py --dataset rby1_20260106_082056

    # 데이터 상세 출력
    python replay_rby1_standalone.py --dataset rby1_20260106_082056 --verbose

    # 특정 프레임 범위 출력
    python replay_rby1_standalone.py --dataset rby1_20260106_082056 --frames 0-10

    # 로봇에 재생 (TODO)
    python replay_rby1_standalone.py --dataset rby1_20260106_082056 --replay --address 192.168.30.1:50051
"""

import argparse
import os
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import rby1_sdk as rby
except ImportError:
    rby = None

# LeRobot 데이터셋
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 그리퍼 방향 설정 (record_rby1_standalone.py와 동일하게 맞출 것)
GRIPPER_DIRECTION = False


# 기본 데이터셋 경로
DEFAULT_DATASETS_DIR = Path.home() / "vla_ws" / "datasets"


def list_datasets(datasets_dir: Path):
    """저장된 데이터셋 목록 출력"""
    if not datasets_dir.exists():
        print(f"데이터셋 폴더가 없습니다: {datasets_dir}")
        return []

    folders = sorted([f for f in datasets_dir.iterdir() if f.is_dir()])
    
    if not folders:
        print(f"저장된 데이터셋이 없습니다: {datasets_dir}")
        return []

    print("=" * 70)
    print(f"저장된 데이터셋 ({datasets_dir})")
    print("=" * 70)
    print(f"{'번호':<4} {'이름':<35} {'크기':<10} {'수정일'}")
    print("-" * 70)

    dataset_names = []
    for i, folder in enumerate(folders):
        # 폴더 크기 계산 (간단히)
        try:
            size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            size_str = format_size(size)
        except:
            size_str = "?"
        
        # 수정 시간
        try:
            mtime = datetime.fromtimestamp(folder.stat().st_mtime)
            mtime_str = mtime.strftime("%Y-%m-%d %H:%M")
        except:
            mtime_str = "?"

        print(f"{i:<4} {folder.name:<35} {size_str:<10} {mtime_str}")
        dataset_names.append(folder.name)

    print("=" * 70)
    return dataset_names


def format_size(size_bytes: int) -> str:
    """바이트를 읽기 좋은 형식으로 변환"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def load_dataset(dataset_name: str, datasets_dir: Path) -> LeRobotDataset:
    """데이터셋 로드"""
    dataset_path = datasets_dir / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {dataset_path}")

    ds = LeRobotDataset(
        repo_id=f"local/{dataset_name}",
        root=dataset_path,
    )
    return ds


def show_dataset_info(ds: LeRobotDataset, dataset_name: str):
    """데이터셋 정보 출력"""
    print("\n" + "=" * 60)
    print(f"데이터셋 정보: {dataset_name}")
    print("=" * 60)
    print(f"  총 프레임: {len(ds)}")
    print(f"  에피소드 수: {ds.num_episodes}")
    print(f"  FPS: {ds.fps}")
    print(f"  로봇 타입: {ds.meta.robot_type}")
    
    # Feature 분류
    obs_keys = [k for k in ds.features.keys() if k.startswith("observation.")]
    action_keys = [k for k in ds.features.keys() if k.startswith("action.")]
    other_keys = [k for k in ds.features.keys() if not k.startswith(("observation.", "action."))]

    print(f"\n  관측 feature ({len(obs_keys)}개):")
    for k in sorted(obs_keys)[:10]:  # 처음 10개만
        print(f"    - {k}")
    if len(obs_keys) > 10:
        print(f"    ... 외 {len(obs_keys) - 10}개")

    print(f"\n  액션 feature ({len(action_keys)}개):")
    for k in sorted(action_keys)[:10]:
        print(f"    - {k}")
    if len(action_keys) > 10:
        print(f"    ... 외 {len(action_keys) - 10}개")

    if other_keys:
        print(f"\n  기타 feature ({len(other_keys)}개):")
        for k in sorted(other_keys):
            print(f"    - {k}")

    print("=" * 60)


def show_frames(ds: LeRobotDataset, start: int, end: int, verbose: bool = False):
    """프레임 데이터 출력"""
    end = min(end, len(ds))
    
    print(f"\n프레임 {start} ~ {end - 1} 출력:")
    print("-" * 60)

    for i in range(start, end):
        frame = ds[i]
        print(f"\n[프레임 {i}]")
        
        if verbose:
            # 전체 데이터 출력
            for k, v in sorted(frame.items()):
                if hasattr(v, "shape"):
                    numel = v.numel() if hasattr(v, "numel") else np.prod(v.shape)
                    if numel <= 10:
                        arr = v.numpy() if hasattr(v, "numpy") else v
                        print(f"  {k}: {arr.flatten()} (shape={v.shape})")
                    else:
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {v}")
        else:
            # 요약 출력
            for k in ["observation.state", "action"]:
                if k in frame:
                    v = frame[k]
                    arr = v.numpy() if hasattr(v, "numpy") else v
                    vals = arr.flatten()
                    fmt = " ".join(f"{x: .4f}" for x in vals)
                    label = "state" if "state" in k else "action"
                    print(f"  {label}: [{fmt}]")
            # 카메라 키 요약
            cam_keys = [k for k in frame.keys() if "images" in k]
            if cam_keys:
                print(f"  cameras: {', '.join(k.split('.')[-1] for k in sorted(cam_keys))}")


def show_camera_images(ds: LeRobotDataset, start: int, end: int, save_images: bool = False, output_dir: Path = None):
    """카메라 이미지 시각화"""
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV가 설치되지 않았습니다: pip install opencv-python")
        return

    # 카메라 feature 찾기
    camera_keys = [k for k in ds.features.keys() if "camera" in k.lower() and ds.features[k].get("dtype") == "video"]
    
    if not camera_keys:
        print("❌ 이 데이터셋에는 카메라 이미지가 없습니다.")
        print(f"   사용 가능한 features: {list(ds.features.keys())[:10]}...")
        return

    print(f"\n카메라 feature 발견: {camera_keys}")
    camera_key = camera_keys[0]  # 첫 번째 카메라 사용
    
    end = min(end, len(ds))
    print(f"프레임 {start} ~ {end - 1} 이미지 표시 (ESC: 종료, SPACE: 다음)")
    
    if save_images and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"이미지 저장 폴더: {output_dir}")

    for i in range(start, end):
        frame = ds[i]
        
        if camera_key not in frame:
            print(f"  프레임 {i}: 카메라 데이터 없음")
            continue
        
        img_tensor = frame[camera_key]
        
        # Tensor to numpy (CHW -> HWC)
        if hasattr(img_tensor, "numpy"):
            img = img_tensor.numpy()
        else:
            img = np.array(img_tensor)
        
        # CHW -> HWC 변환 (필요시)
        if img.shape[0] == 3 and len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # RGB -> BGR (OpenCV용)
        if img.shape[-1] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        # 정보 오버레이
        info_text = f"Frame {i}/{len(ds)-1}"
        cv2.putText(img_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 조인트 위치 표시
        pos_keys = [k for k in frame.keys() if k.endswith(".pos") and "camera" not in k]
        if pos_keys:
            pos_vals = []
            for k in sorted(pos_keys)[:4]:  # 처음 4개만
                v = frame[k]
                val = v.numpy().item() if hasattr(v, "numpy") else float(v)
                pos_vals.append(f"{val:.2f}")
            pos_text = f"Pos: [{', '.join(pos_vals)}...]"
            cv2.putText(img_bgr, pos_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 이미지 저장
        if save_images and output_dir:
            save_path = output_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(save_path), img_bgr)
            print(f"  저장: {save_path.name}")
        
        # 이미지 표시
        cv2.imshow(f"Dataset: {camera_key}", img_bgr)
        
        key = cv2.waitKey(0)  # 키 입력 대기
        if key == 27:  # ESC
            print("\n종료됨")
            break
        elif key == ord(' '):  # SPACE - 다음
            continue
        elif key == ord('s'):  # S - 현재 프레임 저장
            if output_dir is None:
                output_dir = Path.cwd() / "camera_output"
                output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(save_path), img_bgr)
            print(f"  저장: {save_path}")

    cv2.destroyAllWindows()
    print("카메라 뷰어 종료")


def select_dataset_interactive(datasets_dir: Path) -> str | None:
    """대화형으로 데이터셋 선택"""
    dataset_names = list_datasets(datasets_dir)
    
    if not dataset_names:
        return None

    print("\n데이터셋 번호를 입력하세요 (q: 종료): ", end="")
    try:
        choice = input().strip()
        if choice.lower() == "q":
            return None
        
        idx = int(choice)
        if 0 <= idx < len(dataset_names):
            return dataset_names[idx]
        else:
            print(f"잘못된 번호입니다: {idx}")
            return None
    except ValueError:
        # 이름으로 직접 입력한 경우
        if choice in dataset_names:
            return choice
        print(f"잘못된 입력입니다: {choice}")
        return None
    except KeyboardInterrupt:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="RBY1 LeRobot 데이터셋 읽기 및 재생",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 데이터셋 목록 보기
  python replay_rby1_standalone.py --list

  # 대화형 선택
  python replay_rby1_standalone.py

  # 특정 데이터셋 정보
  python replay_rby1_standalone.py --dataset rby1_20260106_082056

  # 프레임 데이터 상세 출력
  python replay_rby1_standalone.py --dataset rby1_20260106_082056 --frames 0-5 --verbose

  # 로봇에 재생
  python replay_rby1_standalone.py --dataset rby1_20260106_082056 --replay --address 192.168.30.1:50051

  # 특정 에피소드, 속도 조절, 그리퍼 포함
  python replay_rby1_standalone.py -d rby1_20260106_082056 --replay --episode 2 --speed 0.5

  # 오른팔만 재생, 그리퍼 없이
  python replay_rby1_standalone.py -d rby1_20260106_082056 --replay --arms right --no-gripper

  # 카메라 이미지 보기
  python replay_rby1_standalone.py --dataset rby1_20260106_082056 --show-camera

  # 특정 프레임의 카메라 이미지 저장
  python replay_rby1_standalone.py --dataset rby1_20260106_082056 --show-camera --frames 0-10 --save-images
        """
    )

    parser.add_argument("--list", action="store_true",
                        help="저장된 데이터셋 목록 출력")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="읽을 데이터셋 이름")
    parser.add_argument("--datasets-dir", type=str, default=None,
                        help=f"데이터셋 폴더 경로 (기본: {DEFAULT_DATASETS_DIR})")
    parser.add_argument("--frames", "-f", type=str, default=None,
                        help="출력할 프레임 범위 (예: 0-10, 5, 0-100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 데이터 출력")
    parser.add_argument("--show-camera", action="store_true",
                        help="카메라 이미지 시각화")
    parser.add_argument("--save-images", action="store_true",
                        help="카메라 이미지 파일로 저장 (--show-camera와 함께 사용)")
    parser.add_argument("--replay", action="store_true",
                        help="로봇에 데이터셋 action 재생")
    parser.add_argument("--address", type=str, default="192.168.30.1:50051",
                        help="로봇 주소 (재생시)")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="재생할 에피소드 번호 (기본: 0)")
    parser.add_argument("--arms", type=str, default="both",
                        choices=["right", "left", "both"],
                        help="재생할 팔 (기본: both)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="재생 속도 배수 (기본: 1.0)")
    parser.add_argument("--no-gripper", action="store_true",
                        help="그리퍼 제어 비활성화")
    parser.add_argument("--source", type=str, default="action",
                        choices=["action", "state"],
                        help="재생 데이터 소스 (기본: action). state=observation.state 사용")
    parser.add_argument("--torso", type=str, default=None,
                        help="초기 torso 자세 설정. 프리셋: 'ready'(45,-90,45), 'packing'(80,-140,60) "
                             "또는 6개 각도(deg) 직접 입력: '0,80,-140,60,0,0'")

    args = parser.parse_args()

    # 데이터셋 경로
    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else DEFAULT_DATASETS_DIR

    # 목록 출력
    if args.list:
        list_datasets(datasets_dir)
        return

    # 데이터셋 선택
    dataset_name = args.dataset
    if dataset_name is None:
        dataset_name = select_dataset_interactive(datasets_dir)
        if dataset_name is None:
            return

    # 데이터셋 로드
    try:
        print(f"\n데이터셋 로드 중: {dataset_name}")
        ds = load_dataset(dataset_name, datasets_dir)
        print("✓ 로드 완료")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return

    # 정보 출력
    show_dataset_info(ds, dataset_name)

    # 프레임 범위 파싱
    start, end = 0, len(ds)
    if args.frames:
        if "-" in args.frames:
            start, end = map(int, args.frames.split("-"))
        else:
            start = int(args.frames)
            end = start + 1

    # 카메라 이미지 시각화
    if args.show_camera:
        output_dir = datasets_dir / dataset_name / "camera_output" if args.save_images else None
        show_camera_images(ds, start, end, save_images=args.save_images, output_dir=output_dir)
        return

    # 프레임 출력
    if args.frames:
        show_frames(ds, start, end, args.verbose)
    elif args.verbose:
        # verbose 모드면 첫 프레임 출력
        show_frames(ds, 0, 1, verbose=True)

    # 재생
    if args.replay:
        # torso 파싱
        torso_pose = None
        if args.torso:
            torso_pose = _parse_torso(args.torso)
            if torso_pose is None:
                print(f"❌ torso 입력 오류: {args.torso}")
                return
        replay_on_robot(
            ds, args.address,
            episode=args.episode,
            arms=args.arms,
            speed=args.speed,
            use_gripper=not args.no_gripper,
            torso_pose=torso_pose,
            source=args.source,
        )
        return


# ============================================================================
# 로봇 재생 기능
# ============================================================================

# Torso 프리셋 (degree)
TORSO_PRESETS = {
    "ready":   [0.0, 45.0, -90.0, 45.0, 0.0, 0.0],
    "packing": [0.0, 80.0, -140.0, 60.0, 0.0, 0.0],
}


def _parse_torso(value: str):
    """
    torso 인자 파싱:
      프리셋 이름 ('ready', 'packing') 또는
      6개 각도(deg) 쉼표 구분 ('0,80,-140,60,0,0')
    반환: np.ndarray (radian, 6개) 또는 None
    """
    key = value.strip().lower()
    if key in TORSO_PRESETS:
        return np.deg2rad(TORSO_PRESETS[key])
    try:
        angles = [float(x.strip()) for x in value.split(",")]
        if len(angles) != 6:
            print(f"❌ torso는 6개 값이 필요합니다 (head_0, torso_0~4). 입력: {len(angles)}개")
            return None
        return np.deg2rad(angles)
    except ValueError:
        return None


class ReplayGripper:
    """재생용 그리퍼 제어 클래스 (record_rby1_standalone.py Gripper와 동일 구조)"""

    def __init__(self):
        self.bus = None
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q = None
        self._running = False
        self._thread = None

    def initialize(self):
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
        if self.bus is None:
            return
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
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

    def set_target(self, normalized_q: np.ndarray):
        """normalized_q: [right, left] 0~1 범위 (record와 동일한 방향)"""
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            return
        normalized_q = np.clip(normalized_q, 0, 1)
        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _control_loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 0.5) for dev_id in [0, 1]])
        while self._running:
            if self.bus and self.target_q is not None:
                try:
                    self.bus.group_sync_write_send_position(
                        [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                    )
                except Exception:
                    pass
            time.sleep(0.1)


def replay_on_robot(ds: LeRobotDataset, address: str, episode: int = 0,
                    arms: str = "both", speed: float = 1.0, use_gripper: bool = True,
                    torso_pose: np.ndarray | None = None, source: str = "action"):
    """
    데이터셋의 action 또는 observation.state를 로봇에 재생

    source: "action" = 마스터 암 명령값(기본), "state" = 로봇 관측값

    action 벡터 구조 (arms="both" 기준):
      [0:7]  = right_arm 관절 (7개)
      [7:14] = left_arm 관절 (7개)
      [14]   = right_gripper (1개)
      [15]   = left_gripper (1개)

    arms="right": [0:7] right_arm + [7] right_gripper
    arms="left":  [0:7] left_arm  + [7] left_gripper
    """
    if rby is None:
        print("❌ rby1_sdk를 찾을 수 없습니다. UPC에서 실행하세요.")
        return

    # ===== 에피소드 추출 =====
    episode_indices = [i for i in range(len(ds)) if ds[i].get("episode_index", -1) == episode]
    if not episode_indices:
        # episode_index 기반 필터가 안 되면 from/to 사용
        if hasattr(ds, 'episode_data_index'):
            ep_from = ds.episode_data_index["from"][episode].item()
            ep_to = ds.episode_data_index["to"][episode].item()
            episode_indices = list(range(ep_from, ep_to))
        else:
            print(f"❌ 에피소드 {episode}를 찾을 수 없습니다.")
            return

    total_frames = len(episode_indices)
    dt = 1.0 / ds.fps  # 프레임 간격
    print(f"\n재생 정보:")
    print(f"  에피소드: {episode}")
    print(f"  프레임 수: {total_frames}")
    print(f"  FPS: {ds.fps}")
    print(f"  예상 시간: {total_frames * dt / speed:.1f}초 (speed={speed}x)")
    print(f"  팔: {arms}")
    print(f"  그리퍼: {'사용' if use_gripper else '미사용'}")
    source_key = "action" if source == "action" else "observation.state"
    print(f"  데이터 소스: {source_key}")

    # ===== 로봇 연결 =====
    print(f"\n로봇 연결 중: {address}")
    robot = rby.create_robot_a(address)
    if not robot.connect():
        print("❌ 로봇 연결 실패")
        return

    if not robot.is_power_on(".*"):
        if not robot.power_on(".*"):
            print("❌ 전원 켜기 실패")
            return
        print("✓ 전원 ON")

    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            print("❌ 서보 활성화 실패")
            return
        print("✓ 서보 ON")

    control_manager_state = robot.get_control_manager_state()
    if control_manager_state.state in (
        rby.ControlManagerState.State.MinorFault,
        rby.ControlManagerState.State.MajorFault,
    ):
        print("⚠ 제어 매니저 fault 감지, 리셋 시도...")
        if not robot.reset_fault_control_manager():
            print("❌ fault 리셋 실패")
            return

    if not robot.enable_control_manager():
        print("❌ 제어 매니저 활성화 실패")
        return
    print("✓ 제어 매니저 활성화")

    # ===== 그리퍼 초기화 (UPC에서만) =====
    gripper = None
    if use_gripper:
        try:
            # 12V 출력 (그리퍼용)
            for arm_name in ["right", "left"]:
                robot.set_tool_flange_output_voltage(arm_name, 12)
            time.sleep(0.5)

            gripper = ReplayGripper()
            if gripper.initialize():
                gripper.homing()
                gripper.start()
            else:
                print("⚠ 그리퍼 없이 진행")
                gripper = None
        except Exception as e:
            print(f"⚠ 그리퍼 초기화 건너뜀: {e}")
            gripper = None

    # ===== 데이터 벡터에서 관절/그리퍼 분리 =====
    # 첫 프레임에서 크기 확인
    first_frame = ds[episode_indices[0]]
    action_tensor = first_frame[source_key]
    action_size = action_tensor.shape[0] if hasattr(action_tensor, "shape") else len(action_tensor)

    # 데이터셋이 양팔(16)인지 한팔(8)인지 판별
    is_both_dataset = (action_size >= 16)  # action=[R7 + L7 + Rgrip + Lgrip]

    # 인덱스 매핑: 양팔 데이터셋이면 항상 올바른 오프셋 사용
    if is_both_dataset:
        right_arm_slice = slice(0, 7)
        left_arm_slice = slice(7, 14)
        r_grip_idx = 14 if action_size > 14 else None
        l_grip_idx = 15 if action_size > 15 else None
    else:
        # 한팔 데이터셋 (action=8)
        right_arm_slice = slice(0, 7)
        left_arm_slice = slice(0, 7)
        r_grip_idx = 7 if (arms in ["right", "both"] and action_size > 7) else None
        l_grip_idx = 7 if (arms in ["left", "both"] and action_size > 7) else None

    if arms == "right":
        has_right_gripper = (r_grip_idx is not None)
        has_left_gripper = False
    elif arms == "left":
        has_right_gripper = False
        has_left_gripper = (l_grip_idx is not None)
    else:  # both
        has_right_gripper = (r_grip_idx is not None)
        has_left_gripper = (l_grip_idx is not None)

    dataset_type = "양팔" if is_both_dataset else "한팔"
    print(f"  action 크기: {action_size} ({dataset_type} 데이터셋)")
    print(f"  재생 팔: {arms} | R그리퍼: {'Y' if has_right_gripper else 'N'}, L그리퍼: {'Y' if has_left_gripper else 'N'}")

    # ===== command stream 생성 =====
    stream = robot.create_command_stream(10)
    # ===== torso 초기 자세 (지정된 경우) =====
    if torso_pose is not None:
        print(f"torso 초기 자세로 이동 중... ({np.rad2deg(torso_pose).round(1)}°)")
        _send_arm_command(stream, torso=torso_pose, minimum_time=5.0)
        time.sleep(5.0)
        print("✓ torso 이동 완료")
    # ===== 첫 프레임으로 이동 (느리게) =====
    first_action = first_frame[source_key].numpy() if hasattr(first_frame[source_key], "numpy") else np.array(first_frame[source_key])
    
    if arms == "right":
        right_arm_pos = first_action[right_arm_slice]
        _send_arm_command(stream, right_arm=right_arm_pos, minimum_time=5.0)
        if gripper and has_right_gripper:
            gripper.set_target(np.array([first_action[r_grip_idx], 0.0]))
    elif arms == "left":
        left_arm_pos = first_action[left_arm_slice]
        _send_arm_command(stream, left_arm=left_arm_pos, minimum_time=5.0)
        if gripper and has_left_gripper:
            gripper.set_target(np.array([0.0, first_action[l_grip_idx]]))
    else:
        right_arm_pos = first_action[right_arm_slice]
        left_arm_pos = first_action[left_arm_slice]
        _send_arm_command(stream, right_arm=right_arm_pos, left_arm=left_arm_pos, minimum_time=5.0)
        if gripper:
            r_grip = first_action[r_grip_idx] if has_right_gripper else 0.0
            l_grip = first_action[l_grip_idx] if has_left_gripper else 0.0
            gripper.set_target(np.array([r_grip, l_grip]))

    print("초기 위치로 이동 중... (5초)")
    time.sleep(5.0)

    # ===== 재생 루프 =====
    print(f"\n▶ 재생 시작! (Ctrl+C로 중단)")
    stopped = False

    def signal_handler(sig, frame):
        nonlocal stopped
        stopped = True
        print("\n⏹ 중단 요청...")

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        for frame_idx, ds_idx in enumerate(episode_indices):
            if stopped:
                break

            frame = ds[ds_idx]
            action = frame[source_key].numpy() if hasattr(frame[source_key], "numpy") else np.array(frame[source_key])

            # 관절 명령 전송
            frame_dt = dt / speed

            # 스트림 만료 시 자동 재생성
            def send_with_retry(**kwargs):
                nonlocal stream
                try:
                    _send_arm_command(stream, **kwargs)
                except RuntimeError as e:
                    if "expired" in str(e).lower():
                        print("\n  ⚠ 스트림 만료, 재생성 중...", end="")
                        stream = robot.create_command_stream(10)
                        _send_arm_command(stream, **kwargs)
                        print(" OK")
                    else:
                        raise

            if arms == "right":
                right_arm_pos = action[right_arm_slice]
                send_with_retry(right_arm=right_arm_pos, minimum_time=frame_dt)
                if gripper and has_right_gripper:
                    gripper.set_target(np.array([action[r_grip_idx], 0.0]))
            elif arms == "left":
                left_arm_pos = action[left_arm_slice]
                send_with_retry(left_arm=left_arm_pos, minimum_time=frame_dt)
                if gripper and has_left_gripper:
                    gripper.set_target(np.array([0.0, action[l_grip_idx]]))
            else:
                right_arm_pos = action[right_arm_slice]
                left_arm_pos = action[left_arm_slice]
                send_with_retry(right_arm=right_arm_pos, left_arm=left_arm_pos, minimum_time=frame_dt)
                if gripper:
                    r_grip = action[r_grip_idx] if has_right_gripper else 0.0
                    l_grip = action[l_grip_idx] if has_left_gripper else 0.0
                    gripper.set_target(np.array([r_grip, l_grip]))

            # 진행률 표시
            if frame_idx % max(1, total_frames // 20) == 0 or frame_idx == total_frames - 1:
                pct = (frame_idx + 1) / total_frames * 100
                elapsed = (frame_idx + 1) * frame_dt
                print(f"  [{pct:5.1f}%] 프레임 {frame_idx}/{total_frames-1} | {elapsed:.1f}s", end="\r")

            time.sleep(frame_dt * 0.95)

        print(f"\n✓ 재생 완료! ({total_frames}프레임)")

    finally:
        signal.signal(signal.SIGINT, old_handler)
        # 정리
        if gripper:
            gripper.stop()
            print("✓ 그리퍼 정지")
        print("✓ 재생 종료")


def _send_arm_command(stream, right_arm=None, left_arm=None, torso=None, minimum_time=0.1):
    """팔/torso 관절 위치 명령 전송 (공식 SDK replay.py 참고)"""
    # control_hold_time: 다음 명령이 올 때까지 현재 위치를 유지하는 시간(초)
    # 데이터셋 프레임 로딩이 느릴 수 있으므로 넉넉하게 설정
    hold_time = max(3.0, minimum_time * 3)
    body_builder = rby.BodyComponentBasedCommandBuilder()

    if torso is not None:
        body_builder.set_torso_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(minimum_time)
            .set_position(torso)
        )

    if right_arm is not None:
        body_builder.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(minimum_time)
            .set_position(right_arm)
        )

    if left_arm is not None:
        body_builder.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(minimum_time)
            .set_position(left_arm)
        )

    rc = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body_builder)
    )
    stream.send_command(rc)


if __name__ == "__main__":
    main()
