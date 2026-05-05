"""Low-level motion primitives."""

from __future__ import annotations

import math

from .utils.frames import normalize_angle


def clamp_pair(left: float, right: float, max_speed: float) -> tuple[float, float]:
    largest = max(abs(left), abs(right), 1e-9)
    if largest <= max_speed:
        return left, right
    scale = max_speed / largest
    return left * scale, right * scale


class MotionController:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def stop(self) -> tuple[float, float]:
        return 0.0, 0.0

    def turn(self, direction: float, speed: float | None = None) -> tuple[float, float]:
        speed = self.cfg.TURN_SPEED if speed is None else speed
        direction = 1.0 if direction >= 0.0 else -1.0
        return clamp_pair(speed * direction, -speed * direction, self.cfg.MAX_WHEEL_SPEED)

    def reverse(self, speed: float | None = None) -> tuple[float, float]:
        speed = self.cfg.REVERSE_SPEED if speed is None else speed
        return clamp_pair(-speed, -speed, self.cfg.MAX_WHEEL_SPEED)

    def arc_forward(self, turn_sign: float = 0.0, speed: float | None = None) -> tuple[float, float]:
        speed = self.cfg.BASE_SPEED if speed is None else speed
        turn = self.cfg.TURN_SPEED * max(-1.0, min(1.0, turn_sign))
        return clamp_pair(speed + turn, speed - turn, self.cfg.MAX_WHEEL_SPEED)

    def follow_path(
        self,
        pose_xy: tuple[float, float],
        yaw_current: float,
        waypoints_world: list[tuple[float, float]],
        waypoint_index: int,
    ) -> tuple[float, float, int, bool]:
        if not waypoints_world:
            return 0.0, 0.0, 0, True

        x_w, z_w = pose_xy
        index = min(max(0, waypoint_index), len(waypoints_world) - 1)
        while index < len(waypoints_world) - 1:
            tx, tz = waypoints_world[index]
            if math.hypot(tx - x_w, tz - z_w) > self.cfg.WAYPOINT_REACHED_DIST_M:
                break
            index += 1

        tx, tz = waypoints_world[index]
        final_x, final_z = waypoints_world[-1]
        if index == len(waypoints_world) - 1 and math.hypot(final_x - x_w, final_z - z_w) <= self.cfg.GOAL_REACHED_DIST_M:
            return 0.0, 0.0, index, True

        lookahead_distance = math.hypot(tx - x_w, tz - z_w)
        while (
            lookahead_distance < self.cfg.PATH_LOOKAHEAD_DIST_M
            and index < len(waypoints_world) - 1
        ):
            index += 1
            tx, tz = waypoints_world[index]
            lookahead_distance = math.hypot(tx - x_w, tz - z_w)

        target_yaw = math.atan2(tz - z_w, tx - x_w)
        yaw_err = normalize_angle(target_yaw - yaw_current)
        if abs(yaw_err) >= self.cfg.PATH_TURN_ONLY_RAD:
            return (*self.turn(yaw_err), index, False)

        turn = max(-1.0, min(1.0, yaw_err / 0.75))
        forward = self.cfg.BASE_SPEED * max(0.30, 1.0 - abs(yaw_err) / math.pi)
        left, right = self.arc_forward(turn, forward)
        return left, right, index, False

    def obstacle_escape(self, ranges: dict[str, float]) -> tuple[float, float] | None:
        front = ranges["front"]
        left = ranges["left"]
        right = ranges["right"]
        if front < self.cfg.FRONT_BLOCK_DIST_M:
            return self.turn(1.0 if left >= right else -1.0, self.cfg.AVOID_TURN_SPEED)
        if left < self.cfg.SIDE_BLOCK_DIST_M:
            return self.arc_forward(-0.45, self.cfg.SLOW_SPEED)
        if right < self.cfg.SIDE_BLOCK_DIST_M:
            return self.arc_forward(0.45, self.cfg.SLOW_SPEED)
        return None
