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
from pathlib import Path
from datetime import datetime

import numpy as np

# LeRobot 데이터셋
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
            # 요약 출력 (관절 위치만)
            pos_keys = [k for k in frame.keys() if k.endswith(".pos")]
            positions = []
            for k in sorted(pos_keys):
                v = frame[k]
                if hasattr(v, "numpy"):
                    positions.append(f"{v.numpy().item():.3f}")
                else:
                    positions.append(f"{v:.3f}")
            
            if positions:
                print(f"  위치: [{', '.join(positions[:7])}]" + (" ..." if len(positions) > 7 else ""))


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
                        help="로봇에 재생 (미구현)")
    parser.add_argument("--address", type=str, default="192.168.30.1:50051",
                        help="로봇 주소 (재생시)")

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

    # 재생 (TODO)
    if args.replay:
        print("\n⚠️  로봇 재생 기능은 아직 구현되지 않았습니다.")
        print("    추후 RBY1 SDK command builder를 사용하여 구현 예정")


if __name__ == "__main__":
    main()
