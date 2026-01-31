#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot 데이터셋 병합 스크립트

같은 task를 수행하는 여러 데이터셋을 하나로 병합합니다.
LeRobot 공식 aggregate.py를 기반으로 로컬 데이터셋용으로 작성되었습니다.

Usage Examples:

모든 open_washer 데이터셋 병합:
    python merge_datasets.py --task open_washer --source_dir 정제버전 --output_dir merged_datasets

특정 데이터셋들만 병합:
    python merge_datasets.py --datasets "rby1_20260116_082157_open_washer,rby1_20260116_085805_open_washer" \
        --output_name merged_open_washer --source_dir 정제버전 --output_dir merged_datasets

모든 task별로 자동 병합:
    python merge_datasets.py --merge_all --source_dir 정제버전 --output_dir merged_datasets
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# lerobot 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

try:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    USE_LEROBOT_AGGREGATE = True
except ImportError:
    USE_LEROBOT_AGGREGATE = False
    logging.warning("LeRobot aggregate module not available. Using standalone implementation.")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_task_from_folder_name(folder_name: str) -> str:
    """폴더 이름에서 task 이름 추출
    
    예: rby1_20260116_082157_open_washer -> open_washer
    """
    parts = folder_name.split("_")
    # rby1_YYYYMMDD_HHMMSS_task_name 형식
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return folder_name


def discover_datasets_by_task(source_dir: Path) -> dict[str, list[Path]]:
    """소스 디렉토리에서 task별로 데이터셋 그룹화
    
    Returns:
        task_name -> [dataset_path, ...]
    """
    datasets_by_task = defaultdict(list)
    
    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith("rby1_"):
            continue
            
        # info.json 확인
        info_path = folder / "meta" / "info.json"
        if not info_path.exists():
            logging.warning(f"Skipping {folder.name}: no meta/info.json found")
            continue
            
        with open(info_path) as f:
            info = json.load(f)
            
        # 유효한 데이터셋인지 확인
        if info.get("total_episodes", 0) == 0:
            logging.warning(f"Skipping {folder.name}: no episodes")
            continue
            
        task = get_task_from_folder_name(folder.name)
        datasets_by_task[task].append(folder)
        
    return dict(datasets_by_task)


def validate_datasets_compatibility(dataset_paths: list[Path]) -> tuple[int, str, dict]:
    """데이터셋들이 병합 가능한지 검증
    
    Returns:
        (fps, robot_type, features)
    """
    fps = None
    robot_type = None
    features = None
    
    for path in dataset_paths:
        info_path = path / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
            
        if fps is None:
            fps = info["fps"]
            robot_type = info["robot_type"]
            features = info["features"]
        else:
            if fps != info["fps"]:
                raise ValueError(f"FPS mismatch: {fps} vs {info['fps']} in {path.name}")
            if robot_type != info["robot_type"]:
                raise ValueError(f"Robot type mismatch: {robot_type} vs {info['robot_type']} in {path.name}")
            # features 비교 (키만 확인)
            if set(features.keys()) != set(info["features"].keys()):
                raise ValueError(f"Features mismatch in {path.name}")
                
    return fps, robot_type, features


def aggregate_stats_manual(all_stats: list[dict]) -> dict:
    """여러 stats를 수동으로 병합"""
    if not all_stats:
        return {}
        
    aggregated = {}
    
    for key in all_stats[0].keys():
        if key not in aggregated:
            aggregated[key] = {}
            
        # 각 통계 항목별로 처리
        for stat_name in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
            values = []
            counts = []
            
            for stats in all_stats:
                if key in stats and stat_name in stats[key]:
                    values.append(np.array(stats[key][stat_name]))
                    if "count" in stats[key]:
                        counts.append(stats[key]["count"][0] if isinstance(stats[key]["count"], list) else stats[key]["count"])
                        
            if not values:
                continue
                
            if stat_name == "min":
                aggregated[key][stat_name] = np.minimum.reduce(values).tolist()
            elif stat_name == "max":
                aggregated[key][stat_name] = np.maximum.reduce(values).tolist()
            elif stat_name == "mean":
                if counts:
                    # 가중 평균
                    total_count = sum(counts)
                    weighted_sum = sum(v * c for v, c in zip(values, counts))
                    aggregated[key][stat_name] = (weighted_sum / total_count).tolist()
                else:
                    aggregated[key][stat_name] = np.mean(values, axis=0).tolist()
            elif stat_name == "std":
                # 근사치 사용 (정확한 계산은 원본 데이터 필요)
                aggregated[key][stat_name] = np.mean(values, axis=0).tolist()
            elif stat_name == "count":
                aggregated[key][stat_name] = [sum(counts)] if counts else values[0]
            elif stat_name.startswith("q"):
                # 분위수는 근사치 사용
                aggregated[key][stat_name] = np.mean(values, axis=0).tolist()
                
    return aggregated


def merge_datasets_standalone(
    dataset_paths: list[Path],
    output_path: Path,
    task_name: str,
) -> None:
    """LeRobot aggregate 없이 수동으로 데이터셋 병합"""
    
    logging.info(f"Merging {len(dataset_paths)} datasets into {output_path}")
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    
    # 검증
    fps, robot_type, features = validate_datasets_compatibility(dataset_paths)
    
    all_data_frames = []
    all_episode_meta = []
    all_stats = []
    video_keys = []
    
    episode_offset = 0
    frame_offset = 0
    
    for src_path in dataset_paths:
        logging.info(f"Processing {src_path.name}")
        
        # info.json 읽기
        with open(src_path / "meta" / "info.json") as f:
            src_info = json.load(f)
            
        # stats.json 읽기
        stats_path = src_path / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                all_stats.append(json.load(f))
                
        # 비디오 키 수집
        for key, feat in src_info.get("features", {}).items():
            if feat.get("dtype") == "video" and key not in video_keys:
                video_keys.append(key)
        
        # parquet 데이터 읽기
        data_dir = src_path / "data" / "chunk-000"
        for parquet_file in sorted(data_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            
            # 인덱스 오프셋 적용
            df["episode_index"] = df["episode_index"] + episode_offset
            df["index"] = df["index"] + frame_offset
            
            all_data_frames.append(df)
            
        # 에피소드 메타데이터 읽기
        episodes_dir = src_path / "meta" / "episodes"
        for ep_file in sorted(episodes_dir.glob("*.parquet")):
            ep_df = pd.read_parquet(ep_file)
            ep_df["episode_index"] = ep_df["episode_index"] + episode_offset
            ep_df["dataset_from_index"] = ep_df["dataset_from_index"] + frame_offset
            ep_df["dataset_to_index"] = ep_df["dataset_to_index"] + frame_offset
            all_episode_meta.append(ep_df)
            
        # 비디오 파일 복사
        videos_dir = src_path / "videos"
        if videos_dir.exists():
            for video_key_dir in videos_dir.iterdir():
                if not video_key_dir.is_dir():
                    continue
                    
                dst_video_dir = output_path / "videos" / video_key_dir.name / "chunk-000"
                dst_video_dir.mkdir(parents=True, exist_ok=True)
                
                for chunk_dir in video_key_dir.iterdir():
                    if not chunk_dir.is_dir():
                        continue
                    for video_file in chunk_dir.glob("*.mp4"):
                        # 새 파일 이름 생성 (에피소드 오프셋 반영)
                        file_idx = int(video_file.stem.split("-")[1])
                        new_file_idx = file_idx + episode_offset
                        dst_file = dst_video_dir / f"file-{new_file_idx:03d}.mp4"
                        
                        if not dst_file.exists():
                            shutil.copy(video_file, dst_file)
                            logging.info(f"  Copied video: {video_file.name} -> {dst_file.name}")
        
        # 오프셋 업데이트
        episode_offset += src_info["total_episodes"]
        frame_offset += src_info["total_frames"]
    
    # 병합된 데이터 저장
    if all_data_frames:
        merged_df = pd.concat(all_data_frames, ignore_index=True)
        merged_df.to_parquet(output_path / "data" / "chunk-000" / "file-000.parquet")
        logging.info(f"Saved merged data: {len(merged_df)} frames")
        
    # 병합된 에피소드 메타데이터 저장
    if all_episode_meta:
        merged_episodes = pd.concat(all_episode_meta, ignore_index=True)
        merged_episodes.to_parquet(output_path / "meta" / "episodes" / "file-000.parquet")
        
    # tasks.parquet 생성
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_name])
    tasks_df.to_parquet(output_path / "meta" / "tasks.parquet")
    
    # info.json 생성
    merged_info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": episode_offset,
        "total_frames": frame_offset,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{episode_offset}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(merged_info, f, indent=4)
        
    # stats.json 병합
    if all_stats:
        merged_stats = aggregate_stats_manual(all_stats)
        with open(output_path / "meta" / "stats.json", "w") as f:
            json.dump(merged_stats, f, indent=4)
            
    # images 폴더 구조 생성 (빈 폴더)
    for video_key in video_keys:
        img_key = video_key.replace("videos/", "images/") if "videos/" in video_key else video_key
        (output_path / "images" / img_key.replace("observation.images.", "observation.images.")).mkdir(
            parents=True, exist_ok=True
        )
    
    logging.info(f"Merge complete: {episode_offset} episodes, {frame_offset} frames")


def merge_with_lerobot_aggregate(
    dataset_paths: list[Path],
    output_path: Path,
    task_name: str,
) -> None:
    """LeRobot 공식 aggregate 함수 사용"""
    
    # repo_id 형식으로 변환 (로컬 경로 사용)
    repo_ids = [f"local/{path.name}" for path in dataset_paths]
    roots = dataset_paths
    
    aggr_repo_id = f"local/{output_path.name}"
    
    try:
        aggregate_datasets(
            repo_ids=repo_ids,
            aggr_repo_id=aggr_repo_id,
            roots=roots,
            aggr_root=output_path,
        )
        logging.info(f"LeRobot aggregate complete: {output_path}")
    except Exception as e:
        logging.error(f"LeRobot aggregate failed: {e}")
        logging.info("Falling back to standalone implementation...")
        merge_datasets_standalone(dataset_paths, output_path, task_name)


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot 데이터셋 병합 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--source_dir",
        type=str,
        default="정제버전",
        help="소스 데이터셋들이 있는 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="merged_datasets",
        help="병합된 데이터셋을 저장할 디렉토리"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="병합할 특정 task 이름 (예: open_washer)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="병합할 데이터셋 폴더명 (쉼표로 구분)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="출력 데이터셋 이름"
    )
    parser.add_argument(
        "--merge_all",
        action="store_true",
        help="모든 task별로 자동 병합"
    )
    parser.add_argument(
        "--use_lerobot",
        action="store_true",
        help="LeRobot 공식 aggregate 함수 사용 (가능한 경우)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실제 병합 없이 어떤 데이터셋이 병합될지만 표시"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        return 1
        
    # task별 데이터셋 탐색
    datasets_by_task = discover_datasets_by_task(source_dir)
    
    if not datasets_by_task:
        logging.error("No valid datasets found")
        return 1
        
    logging.info("Discovered datasets by task:")
    for task, paths in datasets_by_task.items():
        total_episodes = sum(
            json.load(open(p / "meta" / "info.json"))["total_episodes"] 
            for p in paths
        )
        total_frames = sum(
            json.load(open(p / "meta" / "info.json"))["total_frames"] 
            for p in paths
        )
        logging.info(f"  {task}: {len(paths)} datasets, {total_episodes} episodes, {total_frames} frames")
        for p in paths:
            info = json.load(open(p / "meta" / "info.json"))
            logging.info(f"    - {p.name}: {info['total_episodes']} ep, {info['total_frames']} frames")
    
    if args.dry_run:
        logging.info("Dry run complete. No files were modified.")
        return 0
        
    # 병합할 데이터셋 선정
    tasks_to_merge = {}
    
    if args.datasets:
        # 특정 데이터셋들 지정
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        dataset_paths = [source_dir / name for name in dataset_names]
        
        # 존재 확인
        for p in dataset_paths:
            if not p.exists():
                logging.error(f"Dataset not found: {p}")
                return 1
                
        task_name = args.output_name or "custom_merged"
        tasks_to_merge[task_name] = dataset_paths
        
    elif args.task:
        # 특정 task만
        if args.task not in datasets_by_task:
            logging.error(f"Task not found: {args.task}")
            logging.info(f"Available tasks: {list(datasets_by_task.keys())}")
            return 1
        tasks_to_merge[args.task] = datasets_by_task[args.task]
        
    elif args.merge_all:
        # 모든 task
        tasks_to_merge = {
            task: paths for task, paths in datasets_by_task.items()
            if len(paths) > 1  # 2개 이상인 경우만 병합
        }
        if not tasks_to_merge:
            logging.info("No tasks with multiple datasets to merge")
            return 0
    else:
        parser.print_help()
        return 1
    
    # 병합 실행
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task_name, dataset_paths in tasks_to_merge.items():
        if len(dataset_paths) < 2:
            logging.info(f"Skipping {task_name}: only 1 dataset")
            continue
            
        output_name = args.output_name if args.output_name else f"merged_{task_name}"
        output_path = output_dir / output_name
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Merging task: {task_name}")
        logging.info(f"  Datasets: {len(dataset_paths)}")
        logging.info(f"  Output: {output_path}")
        logging.info(f"{'='*60}")
        
        if output_path.exists():
            logging.warning(f"Output path exists, removing: {output_path}")
            shutil.rmtree(output_path)
        
        if args.use_lerobot and USE_LEROBOT_AGGREGATE:
            merge_with_lerobot_aggregate(dataset_paths, output_path, task_name)
        else:
            merge_datasets_standalone(dataset_paths, output_path, task_name)
    
    logging.info("\n" + "="*60)
    logging.info("All merges complete!")
    logging.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
