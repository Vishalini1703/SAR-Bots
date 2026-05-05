"""Configuration for the rebuilt Erebus robot controller."""

from pathlib import Path


def _supervisor_fixed_route_enabled(default: bool = False) -> bool:
    config_path = Path(__file__).resolve().parents[2] / "MainSupervisor" / "config.txt"
    try:
        fields = config_path.read_text(encoding="utf-8").replace("\\", "/").split(",")
    except OSError:
        return default
    if len(fields) <= 7 or fields[7].strip() == "":
        return default
    return bool(int(fields[7]))


MISSION_ROUTE_ENABLED = _supervisor_fixed_route_enabled(default=False)

TIME_STEP_MS = 16

MAX_WHEEL_SPEED = 6.28
BASE_SPEED = 3.4
SLOW_SPEED = 1.4
TURN_SPEED = 2.4
AVOID_TURN_SPEED = 3.4
REVERSE_SPEED = 2.2
LEFT_MOTOR_DIRECTION = 1.0
RIGHT_MOTOR_DIRECTION = 1.0
WHEEL_RADIUS_M = 0.02
WHEEL_TRACK_M = 0.052

COMPASS_PLANAR_AXES = "xy"
USE_GPS_MOTION_YAW = True
GPS_YAW_MIN_DELTA_M = 0.004

SCAN_MAX_RANGE_M = 0.45
FRONT_BLOCK_DIST_M = 0.085
SIDE_BLOCK_DIST_M = 0.045
FRONT_CAUTION_DIST_M = 0.14

MAP_CELL_SIZE_M = 0.03
MAP_WIDTH_CELLS = 60
MAP_HEIGHT_CELLS = 60
MAP_ORIGIN_X_W = -0.90
MAP_ORIGIN_Z_W = -0.90
OBSTACLE_INFLATION_CELLS = 1

WAYPOINT_REACHED_DIST_M = 0.055
GOAL_REACHED_DIST_M = 0.065
PATH_TURN_ONLY_RAD = 0.78
PATH_LOOKAHEAD_DIST_M = 0.13
PATH_REPLAN_INTERVAL_STEPS = 120
PATH_REPLAN_DISTANCE_M = 0.16
PATH_BLOCK_REPLAN_STEPS = 24
FRONTIER_MIN_PATH_CELLS = 2

START_SCAN_STEPS = 155
START_SCAN_TURN_SPEED = 1.7
START_CLEAR_STEPS = 110
START_CLEAR_SPEED = 2.2
START_BACKOUT_STEPS = 55
START_CLEAR_TURN_STEPS = 24
EXPLORE_COMPLETE_FRONTIER_MISSES = 180
EXPLORE_MIN_STEPS_BEFORE_RETURN = 1800
EXPLORE_MAX_STEPS = 7200
EXPLORE_MIN_KNOWN_RATIO = 0.20
PATROL_TURN_STEPS = 18
SUPERVISOR_DEMO_DRIVEN = MISSION_ROUTE_ENABLED
VIDEO_ROUTE_MODE = False
VIDEO_OPEN_LOOP_MODE = False
VIDEO_ROUTE_DRIVE_SPEED = 1.65
VIDEO_ROUTE_TURN_SPEED = 1.85
VIDEO_ROUTE_ARC_GAIN = 0.80
VIDEO_ROUTE_NO_STOP = False
VIDEO_ROUTE_TURN_ONLY_RAD = 0.55
VIDEO_ROUTE_TURN_SCALE = 0.72
VIDEO_ROUTE_MIN_WHEEL_SPEED = 0.35
VIDEO_OPEN_LOOP_DRIVE_SPEED = 2.65
VIDEO_OPEN_LOOP_TURN_SPEED = 2.20
VIDEO_OPEN_LOOP_ARC_SPEED = 2.15
VIDEO_OPEN_LOOP_SCRIPT = (
    ("turn_right", 58),
    ("drive", 450),
    ("turn_left", 58),
    ("drive", 210),
    ("turn_left", 58),
    ("drive", 250),
    ("turn_right", 58),
    ("drive", 360),
    ("turn_right", 58),
    ("drive", 315),
    ("turn_left", 58),
    ("drive", 210),
    ("turn_left", 58),
    ("drive", 360),
    ("turn_left", 58),
    ("drive", 430),
    ("turn_left", 58),
    ("drive", 235),
    ("turn_right", 58),
    ("drive", 330),
    ("turn_right", 58),
    ("drive", 390),
)
FIXED_ROUTE_ENABLED = False
FIXED_ROUTE_LOOP = False
FIXED_ROUTE_REACHED_DIST_M = 0.065
FIXED_ROUTE_TARGET_TIMEOUT_STEPS = 210
FIXED_ROUTE_WAYPOINTS = (
    (-0.48, -0.36),
    (-0.48, -0.22),
    (-0.48, -0.08),
    (-0.38, -0.08),
    (-0.31, 0.06),
    (-0.24, 0.17),
    (-0.30, 0.05),
    (-0.28, -0.11),
    (-0.16, -0.18),
    (-0.04, -0.18),
    (0.08, -0.18),
    (0.16, -0.30),
    (0.30, -0.38),
    (0.46, -0.38),
    (0.34, -0.38),
    (0.20, -0.33),
    (0.08, -0.22),
    (-0.04, -0.18),
    (-0.18, -0.20),
    (-0.31, -0.29),
    (-0.42, -0.40),
    (-0.48, -0.48),
)
DEMO_MISSION_ENABLED = False
DEMO_MISSION_LOOP = False
DEMO_DRIVE_SPEED = 3.2
DEMO_TURN_SPEED = 2.7
DEMO_ARC_SPEED = 2.6
DEMO_RECOVERY_REVERSE_STEPS = 34
DEMO_RECOVERY_TURN_STEPS = 46
DEMO_ESCAPE_ZONE = (-0.40, 0.16, 0.06, 0.52)
DEMO_ESCAPE_TARGET = (0.28, -0.08)
DEMO_ESCAPE_AFTER_STEP = 2500
DEMO_ESCAPE_COOLDOWN_STEPS = 520
DEMO_MISSION_SCRIPT = (
    ("drive", 145),
    ("arc_right", 55),
    ("drive", 175),
    ("turn_left", 45),
    ("drive", 165),
    ("report", "H", -0.234, 0.175),
    ("turn_right", 48),
    ("drive", 150),
    ("arc_left", 70),
    ("drive", 170),
    ("report", "U", 0.065, 0.133),
    ("turn_right", 50),
    ("drive", 230),
    ("arc_right", 75),
    ("drive", 165),
    ("surface", "blue"),
    ("reverse", 80),
    ("turn_right", 110),
    ("drive", 360),
    ("arc_right", 85),
    ("drive", 260),
    ("surface", "black"),
    ("turn_right", 52),
    ("drive", 230),
    ("arc_left", 70),
    ("drive", 150),
    ("report", "S", 0.415, -0.372),
    ("turn_left", 52),
    ("drive", 250),
    ("arc_left", 70),
    ("drive", 220),
)

STUCK_WINDOW_STEPS = 90
STUCK_MIN_PROGRESS_M = 0.012
STUCK_COMMAND_MIN = 0.6
RECOVERY_REVERSE_STEPS = 18
RECOVERY_TURN_STEPS = 26
AVOIDANCE_HOLD_STEPS = 20

WORLD_X_MIN = -0.62
WORLD_X_MAX = 0.62
WORLD_Z_MIN = -0.62
WORLD_Z_MAX = 0.62
WORLD_BOUNDARY_MARGIN_M = 0.035

SWAMP_COST_PENALTY = 3.5
WORLD_TERRAIN_BOUNDS_ENABLED = True
TERRAIN_HAZARD_MARGIN_M = 0.018
TERRAIN_LOOKAHEAD_DIST_M = 0.075
TERRAIN_LOOKAHEAD_SIDE_OFFSET_M = 0.035
TILE_COLOR_CENTER_CROP_RATIO = 0.75
TILE_COLOR_MATCH_MAX_DISTANCE = 0.22
TILE_DARK_MAX_BRIGHTNESS = 0.075
TILE_TRAP_MAX_CHANNEL = 0.075
TILE_TRAP_MAX_CHROMA = 0.035
TILE_TRAP_CONFIRM_STEPS = 2
TILE_COLOR_PROTOTYPES = (
    ("start", "start_tile", (0.0, 0.7, 0.0), 0.0),
    ("checkpoint", "checkpoint_tile_blue", (0.10, 0.10, 0.90), 0.0),
    ("checkpoint", "checkpoint_tile_silver", (0.72, 0.72, 0.68), 0.0),
    ("swamp", "swamp_tile", (0.52, 0.39, 0.19), SWAMP_COST_PENALTY),
    ("special_victim", "special_victim_tile", (0.42, 0.18, 0.70), 0.0),
    ("hazard", "hazard_tile", (0.90, 0.10, 0.10), 0.0),
    ("trap", "hole_tile", (0.02, 0.02, 0.02), 0.0),
)

CAMERA_FOV_RAD = 1.00
CAMERA_MIN_DISTANCE_M = 0.12
CAMERA_MAX_DISTANCE_M = 1.20
VICTIM_WALL_INSET_M = 0.08
CV_MIN_AREA_PX = 45
CV_MAX_AREA_PX = 28000
CV_MIN_ASPECT_RATIO = 0.25
CV_MAX_ASPECT_RATIO = 4.0
CNN_INPUT_SIZE = 32
CNN_LABELS = ("H", "S", "U")
CNN_WEIGHTS_PATH = "models/cnn_shu_weights.npz"
CNN_CONFIRM_N = 3
CNN_CONFIRM_M = 7
DETECTION_MIN_CONFIDENCE = 0.34
DETECTION_SPATIAL_TOLERANCE_M = 0.20
DETECTION_IGNORE_START_RADIUS_M = 0.16
DETECTION_DUPLICATE_RADIUS_M = 0.12
DETECTION_HOLD_STEPS = 22
VICTIM_APPROACH_STANDOFF_M = 0.16
VICTIM_STANDOFF_MAX_RING_CELLS = 8
VICTIM_REPLAN_DISTANCE_M = 0.10
VICTIM_UNREACHABLE_COOLDOWN_STEPS = 500
REPORT_DEDUPE_RADIUS_M = 0.12
REPORT_STOP_STEPS = 70
POST_REPORT_BACKOFF_STEPS = 18
MISSION_VICTIM_TYPES = ("H", "U", "S")
DEFAULT_VICTIM_TYPE = "T"

RETURN_TO_START_STOP_DIST_M = 0.09
EXIT_AFTER_RETURN = True

MISSION_REPORT_PATH = "mission_report.txt"
MISSION_TRAJECTORY_PATH = "trajectory.txt"
MISSION_VICTIMS_PATH = "victim_locations.txt"
MISSION_MAP_PATH = "occupancy_grid.txt"

LOG_TICK_STEPS = False
LOG_DEBUG = False
LOG_FSM_TRANSITIONS = True
MISSION_EVENT_LOGS_ONLY = False
