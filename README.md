# SAR-Bots

An autonomous rescue robot controller built for the **Erebus** simulation environment ([Webots](https://cyberbotics.com/)). The robot navigates a disaster arena, maps its environment, detects victims, and reports their positions — all without human intervention.

---

## Project Structure

```
game/
├── controllers/
│   ├── MainSupervisor/          # Webots supervisor controller (arena management, scoring)
│   │   ├── MainSupervisor.py    # Entry point for the supervisor
│   │   ├── Robot.py             # Robot state tracking
│   │   ├── Tile.py              # Tile/map scoring logic
│   │   ├── Victim.py            # Victim detection and scoring
│   │   ├── MapScorer.py         # Map answer scoring
│   │   ├── MapAnswer.py         # Map answer parsing
│   │   ├── ProtoGenerator.py    # World proto generation
│   │   ├── Camera.py            # Camera utilities
│   │   ├── Logger.py            # Logging utilities
│   │   ├── Config.py            # Supervisor configuration
│   │   └── config.txt           # Runtime config flags
│   │
│   └── robot0Controller/        # Autonomous robot agent
│       ├── robot0Controller.py  # Webots entrypoint
│       └── erebus_controller/   # Core robot intelligence
│           ├── main.py          # Controller bootstrap
│           ├── agent.py         # FSM-based autonomous agent (main logic)
│           ├── config.py        # All tunable parameters
│           ├── robot_io.py      # Sensor/actuator I/O abstraction
│           ├── mapping.py       # Occupancy grid map
│           ├── perception.py    # Victim detection pipeline
│           ├── cnn_classifier.py# Lightweight CNN for victim classification
│           ├── terrain.py       # Floor terrain classification
│           ├── control.py       # Low-level motion controller
│           ├── reporting.py     # Victim report deduplication
│           ├── mission_logging.py# Trajectory and event logging
│           ├── state.py         # Robot state definitions
│           ├── cv_candidates.py # Computer vision candidate detection
│           ├── exploration.py   # Exploration utilities
│           ├── planning/
│           │   └── a_star.py    # A* path planner
│           └── utils/
│               └── frames.py    # Coordinate frame utilities
│
├── worlds/
│   ├── world1.wbt               # Webots simulation world
│   └── textures/                # World textures
│
├── plugins/                     # Webots physics/robot window plugins
└── protos/                      # Custom robot/object PROTO definitions
```

---

## How It Works

The controller is built around a **Finite State Machine (FSM)** with the following states:

| State            | Description                                                       |
|------------------|-------------------------------------------------------------------|
| `START`          | Initial startup scan — robot rotates to build initial map         |
| `EXPLORE`        | Frontier-based autonomous exploration of the arena                |
| `DETECT`         | Victim confirmed — plan approach path                             |
| `APPROACH`       | Navigate toward a detected victim using A*                        |
| `REPORT`         | Stop beside victim and send report to the supervisor              |
| `RETURN_TO_START`| Return to origin after exploration is complete                    |

### Key Subsystems

#### Mapping (`mapping.py`)
- **Occupancy grid** (60×60 cells, 3 cm/cell resolution, ~1.8 m × 1.8 m arena)
- Cells are classified as `UNKNOWN`, `FREE`, `OCCUPIED`, or `HAZARD`
- Obstacle inflation for safe path planning
- Terrain penalties (swamp costs) baked into pathfinding
- Bresenham ray-casting for LIDAR scan integration
- Frontier extraction for exploration targeting

#### Path Planning (`planning/a_star.py`)
- Diagonal-capable **A\* planner** with terrain-aware cell costs
- Replanning triggered by obstacles, terrain hazards, or stuck detection
- Lookahead path following with arc-forward motion

#### Victim Perception (`perception.py`, `cnn_classifier.py`, `cv_candidates.py`)
- Camera captures images; OpenCV-style candidate detection finds bounding boxes
- A **custom CNN** (`TinyCNNClassifier`) classifies wall-mounted victim signs:
  - `H` — Harmed
  - `S` — Stable
  - `U` — Unharmed
- N-of-M confirmation filter prevents false positives
- Spatial deduplication prevents re-reporting the same victim

#### Terrain Classification (`terrain.py`)
- Floor colour sampling classifies tiles:
  - Start tile (green), Checkpoint (blue/silver), Swamp (brown), Hazard (red), Trap/Hole (black)
- Lookahead terrain detection avoids driving into hazards before reaching them

#### Motion Control (`control.py`)
- Differential drive with configurable speeds for forward, arc, turn, and reverse
- Stuck detection via sliding position window with auto-recovery manoeuvre

---

## Configuration

All tunable parameters live in `controllers/robot0Controller/erebus_controller/config.py`.

Key parameters:

| Parameter                  | Default   | Description                              |
|----------------------------|-----------|------------------------------------------|
| `TIME_STEP_MS`             | 16        | Webots simulation timestep (ms)          |
| `BASE_SPEED`               | 3.4       | Normal driving speed (rad/s)             |
| `MAP_CELL_SIZE_M`          | 0.03      | Occupancy grid resolution (m/cell)       |
| `SCAN_MAX_RANGE_M`         | 0.45      | LIDAR max range (m)                      |
| `CNN_LABELS`               | H, S, U   | Victim type labels                       |
| `CNN_WEIGHTS_PATH`         | `models/cnn_shu_weights.npz` | Pre-trained CNN weights   |
| `DETECTION_MIN_CONFIDENCE` | 0.34      | Minimum CNN confidence to accept a detection |
| `EXPLORE_MAX_STEPS`        | 7200      | Max exploration steps before return      |
| `FIXED_ROUTE_ENABLED`      | False     | Enable fixed waypoint route mode         |
| `MISSION_ROUTE_ENABLED`    | (auto)    | Read from supervisor `config.txt`        |

The supervisor's runtime flags are stored in `controllers/MainSupervisor/config.txt`.

---

## Getting Started

### Prerequisites

- **Webots R2023b** or later — [Download](https://cyberbotics.com/#download)
- **Python 3.10+**
- **NumPy** (`pip install numpy`)

### Running the Simulation

1. Open **Webots** and load `worlds/world1.wbt`.
2. Webots will automatically launch both the `MainSupervisor` and `robot0Controller`.
3. The robot will begin its autonomous rescue mission.

### Running the Robot Controller Standalone (for testing)

```bash
# From the repo root — requires Webots' controller module on PYTHONPATH
python controllers/robot0Controller/robot0Controller.py
```

> **Note:** The `controller` module is only available when launched by Webots. Running standalone without it will raise a `RuntimeError` with a helpful message.

---

## Output Files

After a mission run, the following files are generated in the robot controller's working directory:

| File                   | Description                            |
|------------------------|----------------------------------------|
| `mission_report.txt`   | Summary of the mission                 |
| `trajectory.txt`       | Recorded robot positions (x, z, step)  |
| `victim_locations.txt` | Reported victim positions and types    |
| `occupancy_grid.txt`   | Final occupancy grid as ASCII art      |

---

## Competition Modes

The controller supports several operational modes configurable via `config.py`:

| Mode                    | Description                                              |
|-------------------------|----------------------------------------------------------|
| **Autonomous (default)**| Frontier-based exploration with victim detection         |
| `FIXED_ROUTE_ENABLED`   | Follow a pre-defined list of waypoints                   |
| `DEMO_MISSION_ENABLED`  | Scripted demo run with hard-coded actions                |
| `VIDEO_ROUTE_MODE`      | Smooth route-following for video recording               |
| `VIDEO_OPEN_LOOP_MODE`  | Open-loop scripted movement for video                    |

---

## Architecture Overview

```
Webots Robot
     │
     ▼
RobotIO (robot_io.py)        ← sensors, actuators, supervisor messages
     │
     ▼
ErebusAgent (agent.py)       ← FSM orchestrator
  ├── OccupancyGridMap       ← mapping & frontier detection
  ├── AStarPlanner           ← path planning
  ├── MotionController       ← wheel speed commands
  ├── VictimPerceptionPipeline ← camera + CNN victim detection
  ├── TerrainClassifier      ← floor tile classification
  ├── VictimReporter         ← deduplication & reporting
  └── MissionLogger          ← trajectory & event logging
```

---

## License

This project is intended for educational and competition use within the Erebus / RoboCup Rescue simulation framework.
