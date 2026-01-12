# RBY1 LeRobot 데이터 수집 도구

RBY1 로봇을 위한 LeRobot 형식 데이터셋 수집 및 재생 도구입니다.

## 📁 파일 구조

```
vla_ws/
├── record_rby1_standalone.py   # 데이터 녹화 스크립트
├── replay_rby1_standalone.py   # 데이터 확인/재생 스크립트
├── datasets/                   # 저장된 데이터셋
│   └── rby1_YYYYMMDD_HHMMSS/
│       ├── data/               # Parquet 데이터
│       ├── videos/             # 비디오 파일
│       └── meta/               # 메타데이터
├── lerobot/                    # LeRobot 라이브러리
└── rby1-sdk/                   # RBY1 SDK
```

---

## 🎬 데이터 녹화 (record_rby1_standalone.py)

### 기본 사용법

```bash
# 기본 녹화 (5 에피소드, RealSense 카메라 자동 사용)
python record_rby1_standalone.py --address 192.168.30.1:50051 -e 5

# 텔레오퍼레이션 모드 (마스터 암에서 action 기록)
python record_rby1_standalone.py --teleop -e 5

# 일반 USB 카메라 사용 (RealSense 대신)
python record_rby1_standalone.py --no-realsense --camera 0 --teleop -e 5

# 카메라 없이 녹화
python record_rby1_standalone.py --no-realsense -e 5
```

### 인자 설명

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--address` | str | `192.168.30.1:50051` | 로봇 주소 (IP:PORT) |
| `--model` | str | `a` | 로봇 모델: `a`, `m`, `ub` |
| `--arms` | str | `right` | 기록할 팔: `right`, `left`, `both` |
| `--teleop` | flag | false | 텔레오퍼레이션 모드 (마스터 암에서 action 기록) |
| `--camera` | int | None | USB 카메라 ID (예: 0, 1) |
| `--no-realsense` | flag | false | RealSense 카메라 비활성화 (기본: RealSense 사용) |
| `--cameras` | str | auto | 카메라 이름 (쉼표 구분, 예: `cam_high,cam_left_wrist,cam_right_wrist`) |
| `--fps` | int | `30` | 녹화 FPS |
| `--episodes`, `-e` | int | `1` | 녹화할 에피소드 수 |
| `--output` | str | auto | 출력 데이터셋 이름 (기본: `rby1_YYYYMMDD_HHMMSS`) |
| `--task` | str | 프롬프트 | 태스크 설명 (자연어 instruction) |

### 키보드 조작

| 키 | 동작 |
|----|------|
| `SPACE` | 녹화 시작/일시정지 |
| `ENTER` | 현재 에피소드 저장 & 다음 에피소드 |
| `R` | 현재 에피소드 취소 & 다시 녹화 |
| `Q` | 종료 |

### 녹화 모드

| 모드 | 설명 | 사용 시나리오 |
|------|------|---------------|
| **기본 모드** | `action = observation.state` | 외부에서 로봇 조작 시 (SDK 17번 별도 실행) |
| **텔레오프 모드** (`--teleop`) | `action = 마스터 암 위치` | 마스터 암으로 직접 조작하며 녹화 |

---

## 📂 데이터 확인 (replay_rby1_standalone.py)

### 기본 사용법

```bash
# 저장된 데이터셋 목록 보기
python replay_rby1_standalone.py --list

# 데이터셋 정보 확인
python replay_rby1_standalone.py -d rby1_20260107_061029

# 상세 데이터 출력
python replay_rby1_standalone.py -d rby1_20260107_061029 --verbose

# 특정 프레임 범위 출력
python replay_rby1_standalone.py -d rby1_20260107_061029 --frames 0-10
```

### 인자 설명

| 인자 | 설명 |
|------|------|
| `--list` | 저장된 데이터셋 목록 출력 |
| `--dataset`, `-d` | 확인할 데이터셋 이름 |
| `--verbose` | 상세 데이터 출력 |
| `--frames` | 출력할 프레임 범위 (예: `0-10`) |
| `--replay` | 로봇에 재생 (TODO) |

---

## 📊 데이터 형식 (LeRobot 표준)

### 주요 필드

| 필드 | Shape | 설명 |
|------|-------|------|
| `observation.state` | (N,) | 관절+그리퍼 위치 벡터 |
| `action` | (N,) | 목표 위치 벡터 |
| `observation.images.{cam_name}` | (H,W,3) | 카메라 이미지 (멀티 카메라 지원) |
| `observation.velocity` | (14,) | 관절 속도 벡터 |
| `observation.effort` | (14,) | 관절 토크 벡터 |
| `observation.eef_pos` | (12,) | EEF 6D pose (양팔) |
| `task` | string | 자연어 task instruction |

### 벡터 구성 (양팔 기준, 16차원)

```
observation.state / action:
[right_arm_0, right_arm_1, ..., right_arm_6,   # 오른팔 7 관절
 left_arm_0, left_arm_1, ..., left_arm_6,      # 왼팔 7 관절
 right_gripper,                                  # 오른손 그리퍼
 left_gripper]                                   # 왼손 그리퍼
```

### 팔별 차원

| 팔 선택 | 관절 수 | 그리퍼 | 총 차원 | 기본 카메라 |
|---------|---------|--------|---------|-------------|
| `right` | 7 | 1 | 8 | `cam_high`, `cam_right_wrist` (2대) |
| `left` | 7 | 1 | 8 | `cam_high`, `cam_left_wrist` (2대) |
| `both` | 14 | 2 | 16 | `cam_high`, `cam_left_wrist`, `cam_right_wrist` (3대) |

---

## 🔧 설치 요구사항

```bash
# LeRobot 설치
cd lerobot
pip install -e .

# RBY1 SDK 설치
cd rby1-sdk
pip install -e .

# 카메라 (선택)
pip install opencv-python
pip install pyrealsense2  # RealSense 사용시
```

---

## 📝 예시 워크플로우

### 1. 텔레오퍼레이션으로 데이터 수집

```bash
# 마스터 암으로 "컵 집기" 태스크 10 에피소드 수집
python record_rby1_standalone.py \
    --teleop \
    --camera 0 \
    --task "Pick up the red cup and place it on the table" \
    -e 10
```

### 2. 수집된 데이터 확인

```bash
# 데이터셋 목록 확인
python replay_rby1_standalone.py --list

# 상세 데이터 확인
python replay_rby1_standalone.py -d rby1_20260107_123456 --verbose --frames 0-5
```

### 3. 정책 학습 (LeRobot 사용)

```bash
# ACT 정책 학습 예시
python lerobot/scripts/train.py \
    --dataset.repo_id=local/rby1_20260107_123456 \
    --policy.type=act
```

---

## ⚠️ 주의사항

1. **텔레오퍼레이션 모드**는 UPC(Ubuntu PC)에서만 동작합니다
2. **마스터 암**이 연결되어 있어야 `--teleop` 사용 가능
3. 녹화 전 **로봇 파워**가 켜져 있어야 합니다
4. **최대 에피소드 시간**은 60초입니다

---

## 📄 라이센스

이 프로젝트는 연구 목적으로 제공됩니다.
