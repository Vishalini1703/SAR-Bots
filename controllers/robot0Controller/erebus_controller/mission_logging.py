"""Small text outputs for mission review after a Webots run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VictimLogEntry:
    step: int
    victim_type: str
    x_w: float
    z_w: float
    confidence: float


class MissionLogger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.trajectory: list[tuple[int, float, float]] = []
        self.victims: list[VictimLogEntry] = []

    def record_pose(self, step: int, x_w: float, z_w: float) -> None:
        if not self.trajectory or step - self.trajectory[-1][0] >= 8:
            self.trajectory.append((step, x_w, z_w))

    def record_victim(self, step: int, victim_type: str, x_w: float, z_w: float, confidence: float) -> None:
        self.victims.append(VictimLogEntry(step, victim_type, x_w, z_w, confidence))

    def write_summary(self, mission_state: str, start_pose, end_pose, occupancy=None) -> None:
        with open(self.cfg.MISSION_TRAJECTORY_PATH, "w", encoding="utf-8") as file:
            for step, x_w, z_w in self.trajectory:
                file.write(f"{step} {x_w:.4f} {z_w:.4f}\n")

        with open(self.cfg.MISSION_VICTIMS_PATH, "w", encoding="utf-8") as file:
            for item in self.victims:
                file.write(f"{item.step} {item.victim_type} {item.x_w:.4f} {item.z_w:.4f} {item.confidence:.3f}\n")

        with open(self.cfg.MISSION_REPORT_PATH, "w", encoding="utf-8") as file:
            file.write("=== Erebus Mission Report ===\n")
            file.write(f"Final FSM state: {mission_state}\n")
            file.write(f"Trajectory samples: {len(self.trajectory)}\n")
            file.write(f"Victims reported: {len(self.victims)}\n")
            if start_pose is not None:
                file.write(f"Start pose (x,z): ({start_pose.x_w:.4f}, {start_pose.z_w:.4f})\n")
            file.write(f"End pose (x,z): ({end_pose.x_w:.4f}, {end_pose.z_w:.4f})\n")
            if occupancy is not None:
                file.write(f"Explored ratio: {occupancy.explored_ratio():.3f}\n")

        if occupancy is not None:
            with open(self.cfg.MISSION_MAP_PATH, "w", encoding="utf-8") as file:
                for row in occupancy.to_text_rows():
                    file.write(row + "\n")
