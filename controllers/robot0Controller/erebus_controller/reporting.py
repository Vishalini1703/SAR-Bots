"""Victim report de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VictimReport:
    step: int
    timestamp_s: float
    victim_type: str
    x_w: float
    z_w: float
    confidence: float
    confirmation_frames: int


class VictimReporter:
    def __init__(self, dedupe_radius_m: float) -> None:
        self.dedupe_radius_m = float(dedupe_radius_m)
        self._reports: list[VictimReport] = []

    @property
    def reports(self) -> list[VictimReport]:
        return list(self._reports)

    def should_report(self, x_w: float, z_w: float, victim_type: str) -> bool:
        return not self.was_seen(x_w, z_w, victim_type, self.dedupe_radius_m)

    def was_seen(self, x_w: float, z_w: float, victim_type: str, radius_m: float) -> bool:
        for report in self._reports:
            if report.victim_type == victim_type and math.hypot(x_w - report.x_w, z_w - report.z_w) <= radius_m:
                return True
        return False

    def record(
        self,
        step: int,
        timestamp_s: float,
        victim_type: str,
        x_w: float,
        z_w: float,
        confidence: float,
        confirmation_frames: int,
    ) -> VictimReport:
        report = VictimReport(step, timestamp_s, victim_type, x_w, z_w, confidence, confirmation_frames)
        self._reports.append(report)
        return report
