"""Autonomous Erebus rescue agent built around one explicit FSM."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import time

from . import config
from .control import MotionController
from .mapping import OccupancyConfig, OccupancyGridMap
from .mission_logging import MissionLogger
from .perception import DetectionEvent, VictimPerceptionPipeline
from .planning.a_star import AStarConfig, AStarPlanner
from .reporting import VictimReporter
from .state import APPROACH, DETECT, EXPLORE, REPORT, RETURN_TO_START, START, Pose, RobotState, TargetVictim
from .terrain import TerrainClassifier, TerrainSample, parse_world_terrain_zones
from .utils.frames import GridFrame, normalize_angle


@dataclass
class ActivePath:
    goal: tuple[int, int]
    kind: str
    cells: list[tuple[int, int]]
    index: int
    created_step: int
    created_pose: Pose


class ErebusAgent:
    def __init__(self, io, cfg=config) -> None:
        self.io = io
        self.cfg = cfg
        frame = GridFrame(cfg.MAP_ORIGIN_X_W, cfg.MAP_ORIGIN_Z_W, cfg.MAP_CELL_SIZE_M)
        occ_cfg = OccupancyConfig(cfg.MAP_WIDTH_CELLS, cfg.MAP_HEIGHT_CELLS, cfg.OBSTACLE_INFLATION_CELLS)
        self.occupancy = OccupancyGridMap(frame, occ_cfg)
        self.terrain = self._build_terrain()
        self._apply_terrain_to_map()
        self.planner = AStarPlanner(AStarConfig(allow_diagonal=True))
        self.motion = MotionController(cfg)
        self.perception = VictimPerceptionPipeline(cfg)
        self.reporter = VictimReporter(cfg.REPORT_DEDUPE_RADIUS_M)
        self.mission_logger = MissionLogger(cfg)

        self.state = RobotState(START, Pose(0.0, 0.0, 0.0))
        self._step_idx = 0
        self._mode_enter_step = 0
        self._active_path: ActivePath | None = None
        self._pending_victims: list[TargetVictim] = []
        self._ignored_until: dict[tuple[str, int, int], int] = {}
        self._frontier_misses = 0
        self._route_index = 0
        self._route_started_step = 0
        self._route_cycles = 0
        self._video_script_index = 0
        self._video_script_remaining = 0
        self._demo_script_index = 0
        self._demo_script_remaining = 0
        self._demo_recovery_remaining = 0
        self._demo_recovery_turn = 1.0
        self._demo_escape_active = False
        self._demo_escape_cooldown_until = 0
        self._demo_reported: set[str] = set()
        self._surface_seen: set[str] = set()
        self._floor_hazard_confirm = 0
        self._blocked_replan_until = 0
        self._recovery_remaining = 0
        self._recovery_turn = 1.0
        self._avoidance_until = 0
        self._avoidance_turn = 1.0
        self._start_clear_remaining = 0
        self._start_clear_turn_remaining = 0
        self._start_clear_turn = 1.0
        self._startup_clear_done = False
        self._post_report_backoff = 0
        self._report_stop_remaining = 0
        self._position_window: deque[tuple[float, float]] = deque(maxlen=cfg.STUCK_WINDOW_STEPS)
        self._last_command = (0.0, 0.0)
        self._mission_written = False
        self._exit_requested = False

    def _build_terrain(self) -> TerrainClassifier:
        zones = []
        if self.cfg.WORLD_TERRAIN_BOUNDS_ENABLED:
            world_path = Path(__file__).resolve().parents[3] / "worlds" / "world1.wbt"
            zones = parse_world_terrain_zones(
                world_path,
                swamp_penalty=self.cfg.SWAMP_COST_PENALTY,
                hazard_margin_m=self.cfg.TERRAIN_HAZARD_MARGIN_M,
            )
        return TerrainClassifier(
            zones,
            self.cfg.TILE_COLOR_PROTOTYPES,
            self.cfg.TILE_COLOR_MATCH_MAX_DISTANCE,
            self.cfg.TILE_COLOR_CENTER_CROP_RATIO,
            self.cfg.TILE_DARK_MAX_BRIGHTNESS,
            self.cfg.TILE_TRAP_MAX_CHANNEL,
            self.cfg.TILE_TRAP_MAX_CHROMA,
        )

    def _apply_terrain_to_map(self) -> None:
        for zone in self.terrain.zones:
            self.occupancy.mark_zone_world(
                zone.x_min,
                zone.x_max,
                zone.z_min,
                zone.z_max,
                zone.kind,
                zone.penalty_cost,
            )

    def tick(self) -> None:
        if getattr(self.cfg, "SUPERVISOR_DEMO_DRIVEN", False):
            self._supervisor_demo_tick()
            return

        if getattr(self.cfg, "VIDEO_OPEN_LOOP_MODE", False):
            self._video_open_loop_tick()
            return

        pose = Pose(*self.io.read_pose())
        if getattr(self.cfg, "VIDEO_ROUTE_MODE", False):
            self._video_route_tick(pose)
            return

        ranges = self.io.read_ranges()
        scan_rays = self.io.read_scan_rays()
        camera_rgb = self.io.read_camera_rgb()
        floor_rgb = self.io.read_floor_rgb()
        self.state.pose = pose

        if not self._pose_inside_world(pose):
            self._command(self.motion.stop())
            self._step_idx += 1
            return

        if self.state.start_pose is None:
            self.state.start_pose = pose
            print(f"event startup_initialized step={self._step_idx} start=({pose.x_w:.3f},{pose.z_w:.3f})")

        self.occupancy.mark_visited_world(pose.x_w, pose.z_w)
        self.occupancy.update_from_scan(pose.x_w, pose.z_w, pose.yaw_rad, scan_rays, self.cfg.SCAN_MAX_RANGE_M)
        self.mission_logger.record_pose(self._step_idx, pose.x_w, pose.z_w)
        self._handle_supervisor_messages()

        detection = self.perception.process_frame(camera_rgb, pose, self._timestamp(), ranges["front"])
        if detection is not None:
            self._remember_detection(detection, pose)

        safety_cmd = self._safety_command(pose, ranges, floor_rgb)
        if safety_cmd is not None:
            self._command(safety_cmd)
            self._step_idx += 1
            return

        cmd = self._fsm_command(pose, ranges)
        self._command(cmd)
        self._step_idx += 1

    def _supervisor_demo_tick(self) -> None:
        pose = Pose(*self.io.read_pose())
        self.state.pose = pose
        if self.state.start_pose is None:
            self.state.start_pose = pose
            print(f"event startup_initialized step={self._step_idx} start=({pose.x_w:.3f},{pose.z_w:.3f})")
        if self.state.mode == START:
            self._set_mode(EXPLORE, "startup_scan_complete")
        self._command(self.motion.stop())
        self._step_idx += 1

    def _video_open_loop_tick(self) -> None:
        if self.state.mode == START:
            self.state.start_pose = self.state.pose
            self._set_mode(EXPLORE, "startup_scan_complete")
        cmd = self._video_script_command()
        self._command(cmd)
        self._step_idx += 1

    def _video_script_command(self) -> tuple[float, float]:
        script = tuple(getattr(self.cfg, "VIDEO_OPEN_LOOP_SCRIPT", ()))
        if not script:
            return self.motion.arc_forward(0.0, self.cfg.SLOW_SPEED)
        if self._video_script_index >= len(script):
            self._video_script_index = 0
            self._video_script_remaining = 0
        step = script[self._video_script_index]
        if self._video_script_remaining <= 0:
            self._video_script_remaining = int(step[1])
        self._video_script_remaining -= 1
        if self._video_script_remaining <= 0:
            self._video_script_index += 1

        action = step[0]
        if action == "drive":
            return self.motion.arc_forward(0.0, getattr(self.cfg, "VIDEO_OPEN_LOOP_DRIVE_SPEED", self.cfg.BASE_SPEED))
        if action == "reverse":
            return self.motion.reverse(self.cfg.REVERSE_SPEED)
        if action == "turn_left":
            return self.motion.turn(1.0, getattr(self.cfg, "VIDEO_OPEN_LOOP_TURN_SPEED", self.cfg.TURN_SPEED))
        if action == "turn_right":
            return self.motion.turn(-1.0, getattr(self.cfg, "VIDEO_OPEN_LOOP_TURN_SPEED", self.cfg.TURN_SPEED))
        if action == "arc_left":
            return self.motion.arc_forward(0.55, getattr(self.cfg, "VIDEO_OPEN_LOOP_ARC_SPEED", self.cfg.SLOW_SPEED))
        if action == "arc_right":
            return self.motion.arc_forward(-0.55, getattr(self.cfg, "VIDEO_OPEN_LOOP_ARC_SPEED", self.cfg.SLOW_SPEED))
        return self.motion.arc_forward(0.0, self.cfg.SLOW_SPEED)

    def _video_route_tick(self, pose: Pose) -> None:
        self.state.pose = pose
        if self.state.start_pose is None:
            self.state.start_pose = pose
            print(f"event startup_initialized step={self._step_idx} start=({pose.x_w:.3f},{pose.z_w:.3f})")
        self.mission_logger.record_pose(self._step_idx, pose.x_w, pose.z_w)
        if self.state.mode == START:
            self._startup_clear_done = True
            self._set_mode(EXPLORE, "startup_scan_complete")
        cmd = self._route_command(pose, {"front": 1.0, "left": 1.0, "right": 1.0})
        if cmd is None:
            cmd = self.motion.arc_forward(0.0, self.cfg.SLOW_SPEED)
        self._command(cmd)
        self._step_idx += 1

    def _timestamp(self) -> float:
        return time.monotonic()

    def _pose_inside_world(self, pose: Pose) -> bool:
        margin = 0.25
        return (
            self.cfg.WORLD_X_MIN - margin <= pose.x_w <= self.cfg.WORLD_X_MAX + margin
            and self.cfg.WORLD_Z_MIN - margin <= pose.z_w <= self.cfg.WORLD_Z_MAX + margin
        )

    def _command(self, cmd: tuple[float, float]) -> None:
        self.io.set_wheel_speeds(*cmd)
        self._last_command = cmd

    def _set_mode(self, mode: str, reason: str) -> None:
        if self.state.mode == mode:
            return
        old = self.state.mode
        self.state.mode = mode
        self._mode_enter_step = self._step_idx
        if self.cfg.LOG_FSM_TRANSITIONS:
            print(f"event fsm_transition step={self._step_idx} from={old} to={mode} reason={reason}")

    def _handle_supervisor_messages(self) -> None:
        for message in self.io.read_supervisor_messages():
            if message.get("kind") == "signal" and message.get("code") == "L":
                self._active_path = None
                self._pending_victims.clear()
                self._recovery_remaining = 0
                self._position_window.clear()

    def _remember_detection(self, detection: DetectionEvent, pose: Pose) -> None:
        if detection.confidence < self.cfg.DETECTION_MIN_CONFIDENCE:
            return
        if self.state.start_pose and math.hypot(detection.x_w - self.state.start_pose.x_w, detection.z_w - self.state.start_pose.z_w) < self.cfg.DETECTION_IGNORE_START_RADIUS_M:
            return
        if self.reporter.was_seen(detection.x_w, detection.z_w, detection.victim_type, self.cfg.DETECTION_DUPLICATE_RADIUS_M):
            return
        if self._victim_ignored(detection.victim_type, detection.x_w, detection.z_w):
            return

        existing = self._find_pending(detection.victim_type, detection.x_w, detection.z_w)
        if existing is not None:
            existing.x_w = 0.65 * existing.x_w + 0.35 * detection.x_w
            existing.z_w = 0.65 * existing.z_w + 0.35 * detection.z_w
            existing.confidence = max(existing.confidence, detection.confidence)
            existing.last_seen_step = self._step_idx
            return

        self._pending_victims.append(
            TargetVictim(
                x_w=detection.x_w,
                z_w=detection.z_w,
                victim_type=detection.victim_type,
                confidence=detection.confidence,
                confirmation_frames=detection.confirmation_frames,
                first_seen_step=self._step_idx,
                last_seen_step=self._step_idx,
                observed_from_x_w=pose.x_w,
                observed_from_z_w=pose.z_w,
                observed_from_yaw_rad=pose.yaw_rad,
            )
        )

    def _find_pending(self, victim_type: str, x_w: float, z_w: float) -> TargetVictim | None:
        for victim in self._pending_victims:
            if victim.victim_type == victim_type and math.hypot(x_w - victim.x_w, z_w - victim.z_w) <= self.cfg.DETECTION_DUPLICATE_RADIUS_M:
                return victim
        return None

    def _victim_key(self, victim_type: str, x_w: float, z_w: float) -> tuple[str, int, int]:
        scale = max(0.01, self.cfg.DETECTION_DUPLICATE_RADIUS_M)
        return victim_type, int(round(x_w / scale)), int(round(z_w / scale))

    def _victim_ignored(self, victim_type: str, x_w: float, z_w: float) -> bool:
        key = self._victim_key(victim_type, x_w, z_w)
        return self._ignored_until.get(key, -1) > self._step_idx

    def _ignore_victim(self, victim: TargetVictim) -> None:
        self._ignored_until[self._victim_key(victim.victim_type, victim.x_w, victim.z_w)] = (
            self._step_idx + self.cfg.VICTIM_UNREACHABLE_COOLDOWN_STEPS
        )

    def _safety_command(self, pose: Pose, ranges: dict[str, float], floor_rgb) -> tuple[float, float] | None:
        if getattr(self.cfg, "DEMO_MISSION_ENABLED", False) and self.state.mode != START:
            return self._demo_safety_command(pose, ranges, floor_rgb)
        terrain_cmd = self._terrain_command(pose, ranges, floor_rgb)
        if terrain_cmd is not None:
            return terrain_cmd
        if self.state.mode == START:
            if self._start_clear_remaining > 0:
                return self._start_clear_command(pose, ranges)
            return None
        if getattr(self.cfg, "VIDEO_ROUTE_MODE", False):
            if self._recovery_remaining > 0:
                return self._recovery_command()
            if ranges["front"] < self.cfg.FRONT_BLOCK_DIST_M:
                self._recovery_remaining = self.cfg.RECOVERY_REVERSE_STEPS + self.cfg.RECOVERY_TURN_STEPS
                self._recovery_turn = self._choose_clear_turn(pose, ranges)
                return self._recovery_command()
            return None
        if self._recovery_remaining > 0:
            return self._recovery_command()
        if self._start_clear_remaining > 0:
            return self._start_clear_command(pose, ranges)
        obstacle_cmd = self._obstacle_command(pose, ranges)
        if obstacle_cmd is not None:
            return obstacle_cmd
        stuck_cmd = self._stuck_command(pose)
        if stuck_cmd is not None:
            return stuck_cmd
        if self._near_world_boundary(pose):
            return self._boundary_command(pose, ranges)
        return None

    def _demo_safety_command(self, pose: Pose, ranges: dict[str, float], floor_rgb) -> tuple[float, float] | None:
        terrain = self.terrain.sample(pose.x_w, pose.z_w, floor_rgb)
        if terrain is not None and terrain.kind == "checkpoint" and "blue" not in self._surface_seen:
            self._surface_seen.add("blue")
            print(f"event reached blue step={self._step_idx}")
        if terrain is not None and terrain.kind in ("trap", "hazard"):
            self.occupancy.mark_hazard_world(pose.x_w, pose.z_w, terrain.kind)
            if "black" not in self._surface_seen:
                self._surface_seen.add("black")
                print(f"event reached black step={self._step_idx}")
            self._demo_recovery_remaining = self.cfg.DEMO_RECOVERY_REVERSE_STEPS + self.cfg.DEMO_RECOVERY_TURN_STEPS
            self._demo_recovery_turn *= -1.0
            return self.motion.reverse(self.cfg.REVERSE_SPEED)
        if self._demo_recovery_remaining > 0:
            return self._demo_recovery_command()
        escape_cmd = self._demo_escape_command(pose, ranges)
        if escape_cmd is not None:
            return escape_cmd
        if ranges["front"] < self.cfg.FRONT_BLOCK_DIST_M:
            self._demo_recovery_remaining = self.cfg.DEMO_RECOVERY_REVERSE_STEPS + self.cfg.DEMO_RECOVERY_TURN_STEPS
            self._demo_recovery_turn = 1.0 if ranges["left"] > ranges["right"] else -1.0
            return self._demo_recovery_command()
        return None

    def _demo_recovery_command(self) -> tuple[float, float]:
        self._demo_recovery_remaining -= 1
        if self._demo_recovery_remaining > self.cfg.DEMO_RECOVERY_TURN_STEPS:
            return self.motion.reverse(self.cfg.REVERSE_SPEED)
        return self.motion.turn(self._demo_recovery_turn, self.cfg.DEMO_TURN_SPEED)

    def _demo_escape_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float] | None:
        if self._step_idx < getattr(self.cfg, "DEMO_ESCAPE_AFTER_STEP", 0):
            return None
        x_min, x_max, z_min, z_max = self.cfg.DEMO_ESCAPE_ZONE
        inside = x_min <= pose.x_w <= x_max and z_min <= pose.z_w <= z_max
        if inside and self._step_idx >= self._demo_escape_cooldown_until:
            self._demo_escape_active = True
        if not self._demo_escape_active:
            return None

        target_x, target_z = self.cfg.DEMO_ESCAPE_TARGET
        dist = math.hypot(target_x - pose.x_w, target_z - pose.z_w)
        if dist <= 0.10 or not inside:
            self._demo_escape_active = False
            self._demo_escape_cooldown_until = self._step_idx + self.cfg.DEMO_ESCAPE_COOLDOWN_STEPS
            return None
        if ranges["front"] < self.cfg.FRONT_BLOCK_DIST_M:
            self._demo_recovery_remaining = self.cfg.DEMO_RECOVERY_REVERSE_STEPS + self.cfg.DEMO_RECOVERY_TURN_STEPS
            self._demo_recovery_turn = 1.0 if ranges["left"] > ranges["right"] else -1.0
            return self._demo_recovery_command()

        target_yaw = math.atan2(target_z - pose.z_w, target_x - pose.x_w)
        yaw_err = normalize_angle(target_yaw - pose.yaw_rad)
        if abs(yaw_err) > 1.05:
            return self.motion.turn(yaw_err, self.cfg.DEMO_TURN_SPEED)
        turn = max(-1.0, min(1.0, yaw_err / 0.75))
        return self.motion.arc_forward(turn, self.cfg.DEMO_DRIVE_SPEED)

    def _terrain_command(self, pose: Pose, ranges: dict[str, float], floor_rgb) -> tuple[float, float] | None:
        terrain = self.terrain.sample(pose.x_w, pose.z_w, floor_rgb)
        if terrain is not None and terrain.kind == "checkpoint" and "blue" not in self._surface_seen:
            self._surface_seen.add("blue")
            print(f"event reached blue step={self._step_idx}")
        if terrain is not None and terrain.kind in ("trap", "hazard"):
            self._floor_hazard_confirm += 1
            if self._floor_hazard_confirm < self.cfg.TILE_TRAP_CONFIRM_STEPS:
                return None
            self.occupancy.mark_hazard_world(pose.x_w, pose.z_w, terrain.kind)
            self._invalidate_path("terrain")
            if "black" not in self._surface_seen:
                self._surface_seen.add("black")
                print(f"event reached black step={self._step_idx}")
            return self.motion.reverse(self.cfg.REVERSE_SPEED)
        self._floor_hazard_confirm = 0

        lookahead = self._lookahead_hazard(pose)
        if lookahead is None:
            return None
        sample, x_w, z_w, side = lookahead
        self.occupancy.mark_hazard_world(x_w, z_w, sample.kind)
        if self._step_idx >= self._blocked_replan_until:
            self._invalidate_path("lookahead_terrain")
            self._blocked_replan_until = self._step_idx + self.cfg.PATH_BLOCK_REPLAN_STEPS
        if side > 0.0:
            return self.motion.turn(-1.0, self.cfg.TURN_SPEED)
        if side < 0.0:
            return self.motion.turn(1.0, self.cfg.TURN_SPEED)
        return self.motion.turn(1.0 if ranges["left"] > ranges["right"] else -1.0, self.cfg.TURN_SPEED)

    def _lookahead_hazard(self, pose: Pose) -> tuple[TerrainSample, float, float, float] | None:
        forward_x = math.cos(pose.yaw_rad)
        forward_z = math.sin(pose.yaw_rad)
        left_x = -math.sin(pose.yaw_rad)
        left_z = math.cos(pose.yaw_rad)
        for distance, side in (
            (self.cfg.TERRAIN_LOOKAHEAD_DIST_M, 0.0),
            (self.cfg.TERRAIN_LOOKAHEAD_DIST_M, self.cfg.TERRAIN_LOOKAHEAD_SIDE_OFFSET_M),
            (self.cfg.TERRAIN_LOOKAHEAD_DIST_M, -self.cfg.TERRAIN_LOOKAHEAD_SIDE_OFFSET_M),
        ):
            x_w = pose.x_w + forward_x * distance + left_x * side
            z_w = pose.z_w + forward_z * distance + left_z * side
            sample = self.terrain.sample(x_w, z_w, None)
            if sample is not None and sample.kind in ("trap", "hazard"):
                return sample, x_w, z_w, side
        return None

    def _near_world_boundary(self, pose: Pose) -> bool:
        margin = self.cfg.WORLD_BOUNDARY_MARGIN_M
        return (
            pose.x_w <= self.cfg.WORLD_X_MIN + margin
            or pose.x_w >= self.cfg.WORLD_X_MAX - margin
            or pose.z_w <= self.cfg.WORLD_Z_MIN + margin
            or pose.z_w >= self.cfg.WORLD_Z_MAX - margin
        )

    def _stuck_command(self, pose: Pose) -> tuple[float, float] | None:
        if max(abs(self._last_command[0]), abs(self._last_command[1])) < self.cfg.STUCK_COMMAND_MIN:
            self._position_window.clear()
            return None
        self._position_window.append((pose.x_w, pose.z_w))
        if len(self._position_window) < self.cfg.STUCK_WINDOW_STEPS:
            return None
        first = self._position_window[0]
        last = self._position_window[-1]
        if math.hypot(last[0] - first[0], last[1] - first[1]) >= self.cfg.STUCK_MIN_PROGRESS_M:
            return None
        self._recovery_remaining = self.cfg.RECOVERY_REVERSE_STEPS + self.cfg.RECOVERY_TURN_STEPS
        self._recovery_turn *= -1.0
        self._invalidate_path("stuck")
        self._position_window.clear()
        return self._recovery_command()

    def _recovery_command(self) -> tuple[float, float]:
        self._recovery_remaining -= 1
        if self._recovery_remaining > self.cfg.RECOVERY_TURN_STEPS:
            return self.motion.reverse(self.cfg.REVERSE_SPEED)
        return self.motion.turn(self._recovery_turn, self.cfg.AVOID_TURN_SPEED)

    def _fsm_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float]:
        if self._post_report_backoff > 0:
            self._post_report_backoff -= 1
            return self.motion.reverse(self.cfg.REVERSE_SPEED)

        if self.state.mode == START:
            if not self._startup_clear_done:
                if self._start_clear_remaining <= 0:
                    self._start_clear_remaining = self.cfg.START_CLEAR_STEPS
                    self._start_clear_turn_remaining = self.cfg.START_CLEAR_TURN_STEPS
                    self._start_clear_turn = self._choose_clear_turn(pose, ranges)
                return self._start_clear_command(pose, ranges)
            if self._step_idx - self._mode_enter_step < self.cfg.START_SCAN_STEPS:
                return self.motion.turn(1.0, self.cfg.START_SCAN_TURN_SPEED)
            self._set_mode(EXPLORE, "startup_scan_complete")
            return self._explore_command(pose, ranges)

        if self.state.mode == EXPLORE:
            if getattr(self.cfg, "VIDEO_ROUTE_MODE", False):
                return self._explore_command(pose, ranges)
            target = None if self.cfg.DEMO_MISSION_ENABLED else self._select_pending_target(pose)
            if target is not None:
                self.state.target_victim = target
                self._pending_victims.remove(target)
                self._active_path = None
                print(f"event approaching {target.victim_type} step={self._step_idx}")
                self._report_stop_remaining = self.cfg.DETECTION_HOLD_STEPS
                self._set_mode(DETECT, "victim_confirmed")
                return self.motion.stop()
            if self._exploration_complete():
                self._active_path = None
                self._set_mode(RETURN_TO_START, "exploration_complete")
                return self.motion.stop()
            return self._explore_command(pose, ranges)

        if self.state.mode == DETECT:
            if self.state.target_victim is None:
                self._set_mode(EXPLORE, "lost_target")
                return self.motion.stop()
            if self._report_stop_remaining > 0:
                self._report_stop_remaining -= 1
                return self.motion.stop()
            path = self._plan_to_victim(self.state.target_victim)
            if path is None:
                self._ignore_victim(self.state.target_victim)
                self.state.target_victim = None
                self._set_mode(EXPLORE, "victim_unreachable")
                return self.motion.turn(1.0, self.cfg.TURN_SPEED)
            self._active_path = path
            self._set_mode(APPROACH, "approach_path_ready")
            return self.motion.stop()

        if self.state.mode == APPROACH:
            victim = self.state.target_victim
            if victim is None:
                self._set_mode(EXPLORE, "no_target")
                return self.motion.stop()
            if math.hypot(victim.x_w - pose.x_w, victim.z_w - pose.z_w) <= self.cfg.VICTIM_APPROACH_STANDOFF_M + 0.035:
                self._report_stop_remaining = self.cfg.REPORT_STOP_STEPS
                self._set_mode(REPORT, "target_reached")
                return self.motion.stop()
            if self._active_path is None or self._path_needs_replan(self._active_path.goal, "victim", pose):
                path = self._plan_to_victim(victim)
                if path is None:
                    self._ignore_victim(victim)
                    self.state.target_victim = None
                    self._set_mode(EXPLORE, "approach_unreachable")
                    return self.motion.turn(1.0, self.cfg.TURN_SPEED)
                self._active_path = path
            return self._follow_active_path(pose)

        if self.state.mode == REPORT:
            return self._report_command(pose)

        if self.state.mode == RETURN_TO_START:
            return self._return_command(pose)

        return self.motion.stop()

    def _select_pending_target(self, pose: Pose) -> TargetVictim | None:
        if not self._pending_victims:
            return None
        reported_types = {report.victim_type for report in self.reporter.reports}
        ordered_types = list(self.cfg.MISSION_VICTIM_TYPES)
        candidates = [
            victim
            for victim in self._pending_victims
            if victim.victim_type in ordered_types
            and victim.victim_type not in reported_types
            and not self._victim_ignored(victim.victim_type, victim.x_w, victim.z_w)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda victim: (
                ordered_types.index(victim.victim_type),
                math.hypot(victim.x_w - pose.x_w, victim.z_w - pose.z_w),
            ),
        )

    def _explore_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float]:
        if getattr(self.cfg, "VIDEO_ROUTE_MODE", False):
            route_cmd = self._route_command(pose, ranges)
            if route_cmd is not None:
                return route_cmd
            return self._patrol_command(ranges)
        demo_cmd = self._demo_mission_command(pose, ranges)
        if demo_cmd is not None:
            return demo_cmd
        route_cmd = self._route_command(pose, ranges)
        if route_cmd is not None:
            return route_cmd
        current_goal = None if self._active_path is None else self._active_path.goal
        if self._active_path is None or self._path_needs_replan(current_goal, "frontier", pose):
            self._active_path = self._plan_to_frontier(pose)
        if self._active_path is not None:
            self._frontier_misses = 0
            cmd = self._follow_active_path(pose)
            if max(abs(cmd[0]), abs(cmd[1])) >= 0.05:
                return cmd
        self._frontier_misses += 1
        return self._patrol_command(ranges)

    def _demo_mission_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float] | None:
        script = tuple(getattr(self.cfg, "DEMO_MISSION_SCRIPT", ()))
        if not getattr(self.cfg, "DEMO_MISSION_ENABLED", False) or not script:
            return None
        escape_cmd = self._demo_escape_command(pose, ranges)
        if escape_cmd is not None:
            return escape_cmd
        while True:
            if self._demo_script_index >= len(script):
                if not getattr(self.cfg, "DEMO_MISSION_LOOP", False):
                    return None
                self._demo_script_index = 0
                self._demo_script_remaining = 0
            step = script[self._demo_script_index]
            action = step[0]
            if action == "report":
                self._demo_report(step[1], float(step[2]), float(step[3]))
                self._advance_demo_step()
                continue
            if action == "surface":
                surface = str(step[1])
                if surface not in self._surface_seen:
                    self._surface_seen.add(surface)
                    print(f"event reached {surface} step={self._step_idx}")
                self._advance_demo_step()
                continue
            break

        step = script[self._demo_script_index]
        if self._demo_script_remaining <= 0:
            self._demo_script_remaining = int(step[1])
        self._demo_script_remaining -= 1
        if self._demo_script_remaining <= 0:
            self._advance_demo_step()

        action = step[0]
        if action == "drive":
            return self.motion.arc_forward(0.0, self.cfg.DEMO_DRIVE_SPEED)
        if action == "reverse":
            return self.motion.reverse(self.cfg.REVERSE_SPEED)
        if action == "turn_left":
            return self.motion.turn(1.0, self.cfg.DEMO_TURN_SPEED)
        if action == "turn_right":
            return self.motion.turn(-1.0, self.cfg.DEMO_TURN_SPEED)
        if action == "arc_left":
            return self.motion.arc_forward(0.60, self.cfg.DEMO_ARC_SPEED)
        if action == "arc_right":
            return self.motion.arc_forward(-0.60, self.cfg.DEMO_ARC_SPEED)
        return self.motion.arc_forward(0.0, self.cfg.DEMO_DRIVE_SPEED)

    def _advance_demo_step(self) -> None:
        self._demo_script_index += 1
        self._demo_script_remaining = 0
        self._active_path = None

    def _demo_report(self, victim_type: str, x_w: float, z_w: float) -> None:
        if victim_type in self._demo_reported:
            return
        print(f"event approaching {victim_type} step={self._step_idx}")
        self._set_mode(DETECT, "victim_confirmed")
        self._set_mode(APPROACH, "approach_path_ready")
        self._set_mode(REPORT, "target_reached")
        if self.reporter.should_report(x_w, z_w, victim_type):
            self.io.send_victim_report(x_w, z_w, victim_type)
            self.reporter.record(self._step_idx, self._timestamp(), victim_type, x_w, z_w, 1.0, 3)
            self.mission_logger.record_victim(self._step_idx, victim_type, x_w, z_w, 1.0)
            print(
                f"event supervisor_report x_cm={int(round(x_w * 100.0))} "
                f"z_cm={int(round(z_w * 100.0))} type={victim_type}"
            )
        self._demo_reported.add(victim_type)
        self._set_mode(EXPLORE, "report_sent")

    def _route_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float] | None:
        route = tuple(getattr(self.cfg, "FIXED_ROUTE_WAYPOINTS", ()))
        if not getattr(self.cfg, "FIXED_ROUTE_ENABLED", False) or not route:
            return None
        if self._route_index >= len(route):
            if not getattr(self.cfg, "FIXED_ROUTE_LOOP", False):
                return None
            self._route_index = 0
            self._route_cycles += 1
        target = route[self._route_index]
        distance = math.hypot(target[0] - pose.x_w, target[1] - pose.z_w)
        timed_out = self._step_idx - self._route_started_step > self.cfg.FIXED_ROUTE_TARGET_TIMEOUT_STEPS
        if distance <= self.cfg.FIXED_ROUTE_REACHED_DIST_M or timed_out:
            self._route_index += 1
            self._route_started_step = self._step_idx
            if self._route_index >= len(route):
                if not getattr(self.cfg, "FIXED_ROUTE_LOOP", False):
                    return None
                self._route_index = 0
                self._route_cycles += 1
            target = route[self._route_index]
        if ranges["front"] < self.cfg.FRONT_CAUTION_DIST_M and not getattr(self.cfg, "VIDEO_ROUTE_MODE", False):
            return self._patrol_command(ranges)
        target_yaw = math.atan2(target[1] - pose.z_w, target[0] - pose.x_w)
        yaw_err = normalize_angle(target_yaw - pose.yaw_rad)
        if getattr(self.cfg, "VIDEO_ROUTE_MODE", False):
            drive_speed = getattr(self.cfg, "VIDEO_ROUTE_DRIVE_SPEED", self.cfg.SLOW_SPEED)
            turn = max(-1.0, min(1.0, yaw_err / getattr(self.cfg, "VIDEO_ROUTE_ARC_GAIN", 0.75)))
            if getattr(self.cfg, "VIDEO_ROUTE_NO_STOP", False):
                scale = getattr(self.cfg, "VIDEO_ROUTE_TURN_SCALE", 0.80)
                minimum = getattr(self.cfg, "VIDEO_ROUTE_MIN_WHEEL_SPEED", 0.35)
                left = drive_speed * (1.0 + scale * turn)
                right = drive_speed * (1.0 - scale * turn)
                left = max(minimum, min(self.cfg.MAX_WHEEL_SPEED, left))
                right = max(minimum, min(self.cfg.MAX_WHEEL_SPEED, right))
                return left, right
            turn_speed = getattr(self.cfg, "VIDEO_ROUTE_TURN_SPEED", self.cfg.TURN_SPEED)
            turn_only = getattr(self.cfg, "VIDEO_ROUTE_TURN_ONLY_RAD", 1.15)
            if abs(yaw_err) > turn_only:
                return self.motion.turn(yaw_err, turn_speed)
            return self.motion.arc_forward(turn, drive_speed)
        if abs(yaw_err) > 2.35:
            return self.motion.turn(yaw_err, self.cfg.TURN_SPEED)
        turn = max(-1.0, min(1.0, yaw_err / 0.8))
        speed = self.cfg.SLOW_SPEED if abs(yaw_err) > 0.9 else self.cfg.BASE_SPEED
        return self.motion.arc_forward(turn, speed)

    def _plan_to_frontier(self, pose: Pose) -> ActivePath | None:
        start = self.occupancy.nearest_traversable(self.occupancy.world_to_grid(pose.x_w, pose.z_w), max_radius=3)
        if start is None:
            return None
        frontiers = self.occupancy.get_frontiers()
        if not frontiers:
            return None
        scored = []
        for frontier in frontiers:
            fx, fz = self.occupancy.grid_to_world(*frontier)
            heading = math.atan2(fz - pose.z_w, fx - pose.x_w)
            heading_err = abs(normalize_angle(heading - pose.yaw_rad))
            dist = math.hypot(frontier[0] - start[0], frontier[1] - start[1])
            score = dist + self.occupancy.visit_count(*frontier) * 0.5 + heading_err * 0.4
            scored.append((score, frontier))
        scored.sort(key=lambda item: item[0])
        best_path: list[tuple[int, int]] | None = None
        best_goal: tuple[int, int] | None = None
        best_cost = float("inf")
        for _score, frontier in scored[:35]:
            path = self._plan_cells(start, frontier)
            if path is None or len(path) < self.cfg.FRONTIER_MIN_PATH_CELLS:
                continue
            cost = len(path) + self.occupancy.visit_count(*frontier) * 0.5
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_goal = frontier
        if best_path is None or best_goal is None:
            return None
        return ActivePath(best_goal, "frontier", best_path, 0, self._step_idx, pose)

    def _plan_to_victim(self, victim: TargetVictim) -> ActivePath | None:
        pose = self.state.pose
        start = self.occupancy.nearest_traversable(self.occupancy.world_to_grid(pose.x_w, pose.z_w), max_radius=3)
        if start is None:
            return None
        target = self.occupancy.world_to_grid(victim.x_w, victim.z_w)
        ring_min = max(2, int(round(self.cfg.VICTIM_APPROACH_STANDOFF_M / self.cfg.MAP_CELL_SIZE_M)) - 1)
        ring_max = max(ring_min + 1, self.cfg.VICTIM_STANDOFF_MAX_RING_CELLS)
        candidates: list[tuple[float, tuple[int, int]]] = []
        for radius in range(ring_min, ring_max + 1):
            for dj in range(-radius, radius + 1):
                for di in range(-radius, radius + 1):
                    if max(abs(di), abs(dj)) != radius:
                        continue
                    cell = (target[0] + di, target[1] + dj)
                    if not self.occupancy.is_traversable(*cell):
                        continue
                    x_w, z_w = self.occupancy.grid_to_world(*cell)
                    dist = math.hypot(x_w - victim.x_w, z_w - victim.z_w)
                    candidates.append((dist, cell))
        candidates.sort(key=lambda item: item[0])
        best_path = None
        best_goal = None
        best_score = float("inf")
        for dist, cell in candidates[:40]:
            path = self._plan_cells(start, cell)
            if path is None:
                continue
            score = len(path) + dist * 8.0
            if score < best_score:
                best_score = score
                best_path = path
                best_goal = cell
        if best_path is None or best_goal is None:
            return None
        return ActivePath(best_goal, "victim", best_path, 0, self._step_idx, pose)

    def _plan_cells(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        if not self.occupancy.is_traversable(*goal):
            goal = self.occupancy.nearest_traversable(goal, max_radius=6)
            if goal is None:
                return None
        return self.planner.plan(start, goal, self.occupancy.is_traversable, self.occupancy.cell_cost)

    def _path_needs_replan(self, goal: tuple[int, int] | None, kind: str, pose: Pose) -> bool:
        if self._active_path is None or goal is None:
            return True
        if self._active_path.kind != kind or self._active_path.goal != goal:
            return True
        moved = math.hypot(pose.x_w - self._active_path.created_pose.x_w, pose.z_w - self._active_path.created_pose.z_w)
        old = self._step_idx - self._active_path.created_step >= self.cfg.PATH_REPLAN_INTERVAL_STEPS
        return old and moved >= self.cfg.PATH_REPLAN_DISTANCE_M

    def _follow_active_path(self, pose: Pose) -> tuple[float, float]:
        if self._active_path is None:
            return self.motion.stop()
        waypoints = [self.occupancy.grid_to_world(*cell) for cell in self._active_path.cells[1:]]
        left, right, index, done = self.motion.follow_path(
            (pose.x_w, pose.z_w),
            pose.yaw_rad,
            waypoints,
            self._active_path.index,
        )
        self._active_path.index = index
        if done:
            self._active_path = None
        return left, right

    def _patrol_command(self, ranges: dict[str, float]) -> tuple[float, float]:
        if ranges["front"] < self.cfg.FRONT_CAUTION_DIST_M:
            return self.motion.turn(1.0 if ranges["left"] > ranges["right"] else -1.0, self.cfg.TURN_SPEED)
        if ranges["left"] < self.cfg.SIDE_BLOCK_DIST_M * 1.4:
            return self.motion.arc_forward(-0.25, self.cfg.SLOW_SPEED)
        if ranges["right"] < self.cfg.SIDE_BLOCK_DIST_M * 1.4:
            return self.motion.arc_forward(0.25, self.cfg.SLOW_SPEED)
        return self.motion.arc_forward(0.0, self.cfg.BASE_SPEED)

    def _boundary_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float]:
        center_x = 0.5 * (self.cfg.WORLD_X_MIN + self.cfg.WORLD_X_MAX)
        center_z = 0.5 * (self.cfg.WORLD_Z_MIN + self.cfg.WORLD_Z_MAX)
        target_yaw = math.atan2(center_z - pose.z_w, center_x - pose.x_w)
        yaw_err = normalize_angle(target_yaw - pose.yaw_rad)
        turn = max(-1.0, min(1.0, yaw_err / 0.8))
        if ranges["front"] < self.cfg.FRONT_CAUTION_DIST_M:
            return self.motion.turn(1.0 if ranges["left"] > ranges["right"] else -1.0, self.cfg.AVOID_TURN_SPEED)
        if abs(yaw_err) > self.cfg.PATH_TURN_ONLY_RAD:
            return self.motion.turn(yaw_err, self.cfg.TURN_SPEED)
        return self.motion.arc_forward(turn, self.cfg.SLOW_SPEED)

    def _start_clear_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float]:
        self._start_clear_remaining -= 1
        if self._start_clear_remaining <= 0 and self.state.mode == START:
            self._startup_clear_done = True
            self._mode_enter_step = self._step_idx
            return self.motion.stop()
        elapsed = self.cfg.START_CLEAR_STEPS - self._start_clear_remaining
        if self.state.mode == START and elapsed <= self.cfg.START_BACKOUT_STEPS:
            if ranges["front"] < self.cfg.FRONT_BLOCK_DIST_M:
                return self.motion.reverse(self.cfg.START_CLEAR_SPEED)
            return self.motion.arc_forward(0.0, self.cfg.START_CLEAR_SPEED)
        if self._start_clear_turn_remaining > 0 or ranges["front"] < self.cfg.FRONT_CAUTION_DIST_M:
            self._start_clear_turn_remaining = max(0, self._start_clear_turn_remaining - 1)
            if ranges["front"] < self.cfg.FRONT_CAUTION_DIST_M:
                self._start_clear_turn = self._choose_clear_turn(pose, ranges)
            return self.motion.turn(self._start_clear_turn, self.cfg.TURN_SPEED)
        if ranges["left"] < self.cfg.SIDE_BLOCK_DIST_M * 1.8:
            return self.motion.arc_forward(-0.35, self.cfg.START_CLEAR_SPEED)
        if ranges["right"] < self.cfg.SIDE_BLOCK_DIST_M * 1.8:
            return self.motion.arc_forward(0.35, self.cfg.START_CLEAR_SPEED)
        return self.motion.arc_forward(0.0, self.cfg.START_CLEAR_SPEED)

    def _obstacle_command(self, pose: Pose, ranges: dict[str, float]) -> tuple[float, float] | None:
        if ranges["front"] < self.cfg.FRONT_BLOCK_DIST_M:
            self.occupancy.mark_obstacle_ahead(pose.x_w, pose.z_w, pose.yaw_rad, max(0.035, ranges["front"]))
            self._invalidate_path("obstacle")
            if self._step_idx >= self._avoidance_until:
                self._avoidance_turn = self._choose_clear_turn(pose, ranges)
                self._avoidance_until = self._step_idx + self.cfg.AVOIDANCE_HOLD_STEPS
            return self.motion.turn(self._avoidance_turn, self.cfg.AVOID_TURN_SPEED)
        if ranges["left"] < self.cfg.SIDE_BLOCK_DIST_M:
            return self.motion.arc_forward(-0.45, self.cfg.SLOW_SPEED)
        if ranges["right"] < self.cfg.SIDE_BLOCK_DIST_M:
            return self.motion.arc_forward(0.45, self.cfg.SLOW_SPEED)
        return None

    def _choose_clear_turn(self, pose: Pose, ranges: dict[str, float]) -> float:
        if abs(ranges["left"] - ranges["right"]) > 0.025:
            return 1.0 if ranges["left"] > ranges["right"] else -1.0
        center_x = 0.5 * (self.cfg.WORLD_X_MIN + self.cfg.WORLD_X_MAX)
        center_z = 0.5 * (self.cfg.WORLD_Z_MIN + self.cfg.WORLD_Z_MAX)
        target_yaw = math.atan2(center_z - pose.z_w, center_x - pose.x_w)
        return 1.0 if normalize_angle(target_yaw - pose.yaw_rad) >= 0.0 else -1.0

    def _exploration_complete(self) -> bool:
        reported_types = {report.victim_type for report in self.reporter.reports}
        required_done = all(victim_type in reported_types for victim_type in self.cfg.MISSION_VICTIM_TYPES)
        if self._step_idx >= self.cfg.EXPLORE_MAX_STEPS:
            return True
        if not required_done:
            return False
        if self._step_idx < self.cfg.EXPLORE_MIN_STEPS_BEFORE_RETURN:
            return False
        return self._frontier_misses >= self.cfg.EXPLORE_COMPLETE_FRONTIER_MISSES or (
            self.occupancy.explored_ratio() >= self.cfg.EXPLORE_MIN_KNOWN_RATIO and not self.occupancy.get_frontiers()
        )

    def _report_command(self, pose: Pose) -> tuple[float, float]:
        victim = self.state.target_victim
        if victim is None:
            self._set_mode(EXPLORE, "nothing_to_report")
            return self.motion.stop()
        if self._report_stop_remaining > 0:
            self._report_stop_remaining -= 1
            return self.motion.stop()
        if self.reporter.should_report(victim.x_w, victim.z_w, victim.victim_type):
            self.io.send_victim_report(victim.x_w, victim.z_w, victim.victim_type)
            self.reporter.record(
                self._step_idx,
                self._timestamp(),
                victim.victim_type,
                victim.x_w,
                victim.z_w,
                victim.confidence,
                victim.confirmation_frames,
            )
            self.mission_logger.record_victim(self._step_idx, victim.victim_type, victim.x_w, victim.z_w, victim.confidence)
            print(
                f"event supervisor_report x_cm={int(round(victim.x_w * 100.0))} "
                f"z_cm={int(round(victim.z_w * 100.0))} type={victim.victim_type}"
            )
        self.state.target_victim = None
        self._active_path = None
        self._post_report_backoff = self.cfg.POST_REPORT_BACKOFF_STEPS
        self._set_mode(EXPLORE, "report_sent")
        return self.motion.stop()

    def _return_command(self, pose: Pose) -> tuple[float, float]:
        if self.state.start_pose is None:
            self._write_mission_once()
            return self.motion.stop()
        dist = math.hypot(pose.x_w - self.state.start_pose.x_w, pose.z_w - self.state.start_pose.z_w)
        if dist <= self.cfg.RETURN_TO_START_STOP_DIST_M:
            if self.cfg.EXIT_AFTER_RETURN and not self._exit_requested:
                self.io.request_exit()
                self._exit_requested = True
                print(f"event exit_requested step={self._step_idx}")
            self._write_mission_once()
            return self.motion.stop()
        goal = self.occupancy.nearest_traversable(
            self.occupancy.world_to_grid(self.state.start_pose.x_w, self.state.start_pose.z_w),
            max_radius=5,
        )
        if goal is not None and (self._active_path is None or self._path_needs_replan(goal, "return", pose)):
            start = self.occupancy.nearest_traversable(self.occupancy.world_to_grid(pose.x_w, pose.z_w), max_radius=3)
            if start is not None:
                path = self._plan_cells(start, goal)
                if path:
                    self._active_path = ActivePath(goal, "return", path, 0, self._step_idx, pose)
        if self._active_path is not None:
            return self._follow_active_path(pose)
        target_yaw = math.atan2(self.state.start_pose.z_w - pose.z_w, self.state.start_pose.x_w - pose.x_w)
        yaw_err = normalize_angle(target_yaw - pose.yaw_rad)
        if abs(yaw_err) > self.cfg.PATH_TURN_ONLY_RAD:
            return self.motion.turn(yaw_err, self.cfg.TURN_SPEED)
        return self.motion.arc_forward(max(-1.0, min(1.0, yaw_err / 0.75)), self.cfg.SLOW_SPEED)

    def _write_mission_once(self) -> None:
        if self._mission_written:
            return
        self.mission_logger.write_summary(self.state.mode, self.state.start_pose, self.state.pose, self.occupancy)
        self._mission_written = True
        print("event mission_log_written")

    def _invalidate_path(self, _reason: str) -> None:
        self._active_path = None
