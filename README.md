# RBY1 LeRobot Data Collection Tool

Data collection and replay tool for LeRobot format datasets on RBY1 robot.

## 📁 File Structure

```
rby1-lerobot_dataset/
├── record_rby1_standalone.py   # Data recording script (main)
├── replay_rby1_standalone.py   # Data inspection/replay script
├── merge_datasets.py           # Dataset merging utility
├── datasets/                   # Saved datasets
│   └── rby1_YYYYMMDD_HHMMSS/
│       ├── data/               # Parquet data
│       ├── videos/             # Video files
│       └── meta/               # Metadata
├── lerobot/                    # LeRobot library
└── rby1-sdk/                   # RBY1 SDK
```

---

## 🎬 Data Recording (record_rby1_standalone.py)

### Two Operating Modes

| Mode | Description | Control Authority | Use Case |
|------|-------------|-------------------|----------|
| **Observation-only** (default) | Read state and record only | ❌ None | Run alongside SDK teleoperation |
| **Teleoperation** (`--teleop`) | Control robot with master arm + record | ✅ Acquired | Standalone execution |

### Basic Usage

```bash
# === Observation-only Mode (use with SDK teleoperation) ===
# Terminal 1: Run SDK teleoperation (robot control)
python rby1-sdk/examples/python/99_teleoperation_with_joint_mapping.py --address 192.168.30.1:50051

# Terminal 2: Recording only (no control authority)
python record_rby1_standalone.py --address 192.168.30.1:50051 -e 5

# === Teleoperation Mode (standalone) ===
# Control robot directly with master arm while recording
python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop -e 5

# Teleoperation with impedance control (default, compliant)
python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop -e 5

# Teleoperation with position control (precise)
python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --mode position -e 5

# Recording without camera
python record_rby1_standalone.py --address 192.168.30.1:50051 --no-realsense -e 5

# With camera web streaming (view at http://localhost:8000)
python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --stream 8000 -e 5

# Disable pose reset at each episode start
python record_rby1_standalone.py --address 192.168.30.1:50051 --teleop --no-reset -e 5
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--address` | str | `192.168.30.1:50051` | Robot address (IP:PORT) |
| `--model` | str | `a` | Robot model: `a`, `m`, `ub` |
| `--arms` | str | `right` | Arms to record: `right`, `left`, `both` |
| `--teleop` | flag | false | **Teleoperation mode** (acquire control, control robot with master arm) |
| `--mode` | str | `impedance` | Control mode: `position` (precise) or `impedance` (compliant) |
| `--no-reset` | flag | false | Disable pose reset at each episode start |
| `--camera` | int | None | USB camera ID (e.g., 0, 1) |
| `--no-realsense` | flag | false | Disable RealSense camera (default: RealSense enabled) |
| `--cameras` | str | auto | Camera names (comma-separated, e.g., `cam_high,cam_left_wrist,cam_right_wrist`) |
| `--stream` | int | `0` | Camera web streaming port (e.g., `8000`, 0 to disable) |
| `--fps` | int | `30` | Recording FPS |
| `--episodes`, `-e` | int | `1` | Number of episodes to record |
| `--output` | str | auto | Output dataset name (default: `rby1_YYYYMMDD_HHMMSS`) |
| `--task` | str | prompt | Task description (natural language instruction) |
| `--wheels` | flag | false | **[DEV]** Enable wheel data recording |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start/pause recording |
| `ENTER` | Save current episode & next episode |
| `R` | Cancel current episode & re-record |
| `B` | Delete previous episode & re-record |
| `Q` | Quit |

---

## 📂 Data Inspection (replay_rby1_standalone.py)

### Basic Usage

```bash
# List saved datasets
python replay_rby1_standalone.py --list

# Inspect dataset info
python replay_rby1_standalone.py -d rby1_20260107_061029

# Verbose data output
python replay_rby1_standalone.py -d rby1_20260107_061029 --verbose

# Specific frame range
python replay_rby1_standalone.py -d rby1_20260107_061029 --frames 0-10
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--list` | List saved datasets |
| `--dataset`, `-d` | Dataset name to inspect |
| `--verbose` | Verbose data output |
| `--frames` | Frame range to output (e.g., `0-10`) |
| `--replay` | Replay on robot (TODO) |

---

## 📊 Data Format (LeRobot Standard)

### Main Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `observation.state` | (N,) | Joint + gripper position vector |
| `action` | (N,) | Target position vector |
| `observation.images.{cam_name}` | (H,W,3) | Camera images (multi-camera support) |
| `observation.velocity` | (14,) | Joint velocity vector |
| `observation.effort` | (14,) | Joint torque vector |
| `observation.eef_pos` | (12,) | EEF 6D pose (both arms) |
| `task` | string | Natural language task instruction |

### Vector Structure (both arms, 16 dimensions)

```
observation.state / action:
[right_arm_0, right_arm_1, ..., right_arm_6,   # Right arm 7 joints
 left_arm_0, left_arm_1, ..., left_arm_6,      # Left arm 7 joints
 right_gripper,                                 # Right gripper
 left_gripper]                                  # Left gripper
```

### Dimensions by Arm Selection

| Arm Selection | Joints | Gripper | Total Dim | Default Cameras |
|---------------|--------|---------|-----------|-----------------|
| `right` | 7 | 1 | 8 | `cam_high`, `cam_right_wrist` (2) |
| `left` | 7 | 1 | 8 | `cam_high`, `cam_left_wrist` (2) |
| `both` | 14 | 2 | 16 | `cam_high`, `cam_left_wrist`, `cam_right_wrist` (3) |

---

## 🔧 Installation Requirements

```bash
# Install LeRobot
cd lerobot
pip install -e .

# Install RBY1 SDK
cd rby1-sdk
pip install -e .

# Camera (optional)
pip install opencv-python
pip install pyrealsense2  # For RealSense
```

---

## 📝 Example Workflow

### 1. Collect Data via Teleoperation

```bash
# Collect 10 episodes of "pick up cup" task with master arm
python record_rby1_standalone.py \
    --teleop \
    --task "Pick up the red cup and place it on the table" \
    -e 10
```

### 2. Inspect Collected Data

```bash
# List datasets
python replay_rby1_standalone.py --list

# Inspect data in detail
python replay_rby1_standalone.py -d rby1_20260107_123456 --verbose --frames 0-5
```

### 3. Train Policy (using LeRobot)

```bash
# ACT policy training example
python lerobot/scripts/train.py \
    --dataset.repo_id=local/rby1_20260107_123456 \
    --policy.type=act
```

---

## ⚠️ Notes

1. **Observation-only Mode** (default):
   - Does not acquire control authority, **can run alongside SDK teleoperation**
   - Run SDK teleoperation first, then run the recording script
   
2. **Teleoperation Mode** (`--teleop`):
   - Acquires control authority, **cannot run alongside SDK teleoperation**
   - Only works on UPC (Ubuntu PC)
   - Requires master arm connection

3. **Control Mode**:
   - `impedance` (default): Compliant control, safe for human interaction
   - `position`: Precise position control, stiffer response

4. **Pose Reset**: Robot returns to ready pose at each episode start (disable with `--no-reset`)

5. **Robot power** must be on before recording
   - Observation-only mode: SDK teleoperation already turns it on
   - Teleoperation mode: Script automatically powers on

6. **Maximum episode duration** is 5 minutes

7. **RealSense camera** is enabled by default (disable with `--no-realsense`)

8. **Camera streaming**: Use `--stream 8000` to view cameras at `http://localhost:8000`

---

## � Merging Datasets (merge_datasets.py)

Merge multiple datasets with the same task description into a single dataset.

```bash
# Merge all datasets containing "pick" in task description
python merge_datasets.py --task pick --output merged_pick_dataset

# Merge specific datasets by name
python merge_datasets.py --datasets rby1_20260107_123456 rby1_20260108_234567 --output merged_dataset
```

---

## �🚧 Development Features

| Feature | Flag | Description |
|---------|------|-------------|
| Wheel recording | `--wheels` | Record wheel encoder data (wheel_0, wheel_1) |

---

## 📄 License

This project is provided for research purposes.

## ⚖️ Disclaimer

This software is provided "AS-IS" without any warranties. The authors and contributors assume no responsibility for any damages, injuries, or equipment malfunctions resulting from the use of this software. Use at your own risk.
