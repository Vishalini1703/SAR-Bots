"""FSM state containers."""

from __future__ import annotations

from dataclasses import dataclass

START = "START"
EXPLORE = "EXPLORE"
DETECT = "DETECT"
APPROACH = "APPROACH"
REPORT = "REPORT"
RETURN_TO_START = "RETURN_TO_START"


@dataclass(frozen=True)
class Pose:
    x_w: float
    z_w: float
    yaw_rad: float


@dataclass
class TargetVictim:
    x_w: float
    z_w: float
    victim_type: str
    confidence: float
    confirmation_frames: int
    first_seen_step: int
    last_seen_step: int
    observed_from_x_w: float
    observed_from_z_w: float
    observed_from_yaw_rad: float


@dataclass
class RobotState:
    mode: str
    pose: Pose
    start_pose: Pose | None = None
    target_victim: TargetVictim | None = None
