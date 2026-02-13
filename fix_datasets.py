#!/usr/bin/env python3
"""
images만 있고 videos가 없는 데이터셋에 비디오를 생성하는 스크립트
"""

import os
import sys
import json
from pathlib import Path

# lerobot 경로 추가
sys.path.insert(0, str(Path.home() / "vla_ws" / "lerobot" / "src"))

from lerobot.datasets.video_utils import encode_video_frames

def fix_dataset(dataset_path: Path):
    """단일 데이터셋 수정"""
    images_dir = dataset_path / "images"
    videos_dir = dataset_path / "videos"
    meta_dir = dataset_path / "meta"
    
    if not images_dir.exists():
        print(f"  ❌ images 폴더 없음: {dataset_path.name}")
        return False
    
    if videos_dir.exists():
        print(f"  ⏭️ videos 이미 존재: {dataset_path.name}")
        return True
    
    # info.json에서 fps 읽기
    info_path = meta_dir / "info.json"
    fps = 30  # 기본값
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            fps = info.get("fps", 30)
    
    print(f"  🎬 비디오 생성 중... (fps={fps})")
    
    # 각 카메라별로 비디오 생성
    for cam_dir in images_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        
        cam_name = cam_dir.name  # e.g., observation.images.cam_high
        
        # 에피소드별로 처리
        for episode_dir in sorted(cam_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            
            episode_name = episode_dir.name  # e.g., episode-000000
            
            # 프레임 수 확인
            frames = list(episode_dir.glob("frame-*.png"))
            if len(frames) == 0:
                print(f"    ⚠️ 프레임 없음: {cam_name}/{episode_name}")
                continue
            
            # 비디오 출력 경로
            # LeRobot 형식: videos/{cam_name}/chunk-000/file-000.mp4
            chunk_idx = 0  # 첫 번째 청크
            file_idx = int(episode_name.split("-")[1])  # episode 번호가 file 번호
            
            video_out_dir = videos_dir / cam_name / f"chunk-{chunk_idx:03d}"
            video_out_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_out_dir / f"file-{file_idx:03d}.mp4"
            
            print(f"    📹 {cam_name}/{episode_name} ({len(frames)} frames) -> {video_path.name}")
            
            try:
                encode_video_frames(
                    imgs_dir=episode_dir,
                    video_path=video_path,
                    fps=fps,
                    vcodec="h264",  # 호환성 좋은 코덱 사용
                    pix_fmt="yuv420p",
                    g=2,
                    crf=23,
                    overwrite=True
                )
            except Exception as e:
                print(f"    ❌ 인코딩 실패: {e}")
                return False
    
    print(f"  ✅ 완료: {dataset_path.name}")
    return True


def main():
    datasets_dir = Path.home() / "vla_ws" / "datasets"
    
    # images만 있고 videos가 없는 폴더 찾기
    to_fix = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        if d.name.endswith(".tar.gz"):
            continue
            
        images_dir = d / "images"
        videos_dir = d / "videos"
        
        if images_dir.exists() and not videos_dir.exists():
            to_fix.append(d)
    
    print(f"수정할 데이터셋: {len(to_fix)}개\n")
    
    success = 0
    failed = 0
    
    for i, dataset_path in enumerate(to_fix, 1):
        print(f"[{i}/{len(to_fix)}] {dataset_path.name}")
        if fix_dataset(dataset_path):
            success += 1
        else:
            failed += 1
        print()
    
    print(f"\n{'='*50}")
    print(f"완료: {success}개 성공, {failed}개 실패")


if __name__ == "__main__":
    main()
