"""Perception pipeline: CV candidates followed by CNN confirmation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics

import numpy as np

from .cnn_classifier import TinyCNNClassifier
from .cv_candidates import CVCandidateGenerator
from .state import Pose
from .utils.frames import normalize_angle


@dataclass(frozen=True)
class DetectionEvent:
    x_w: float
    z_w: float
    victim_type: str
    confidence: float
    confirmation_frames: int
    timestamp_s: float


class VictimPerceptionPipeline:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.candidates = CVCandidateGenerator(
            cfg.CV_MIN_AREA_PX,
            cfg.CV_MAX_AREA_PX,
            cfg.CV_MIN_ASPECT_RATIO,
            cfg.CV_MAX_ASPECT_RATIO,
        )
        self.classifier = TinyCNNClassifier(cfg.CNN_INPUT_SIZE, tuple(cfg.CNN_LABELS), cfg.CNN_WEIGHTS_PATH)
        self.history: deque[tuple[str, float, float, float]] = deque(maxlen=cfg.CNN_CONFIRM_M)

    def process_frame(
        self,
        rgb_frame: np.ndarray | None,
        pose: Pose,
        timestamp_s: float,
        front_range_m: float,
    ) -> DetectionEvent | None:
        if rgb_frame is None or rgb_frame.size == 0:
            return None
        frame_h, frame_w = rgb_frame.shape[:2]
        best: tuple[float, str, float, float] | None = None
        for box in self.candidates.generate(rgb_frame):
            crop = rgb_frame[box.y : box.y + box.h, box.x : box.x + box.w]
            prediction = self.classifier.predict(crop)
            score = prediction.confidence * max(0.25, box.score)
            if score < self.cfg.DETECTION_MIN_CONFIDENCE:
                continue
            center_x = box.x + box.w * 0.5
            x_w, z_w = self._estimate_world(pose, center_x, box.w * box.h, frame_w * frame_h, frame_w, front_range_m)
            if best is None or score > best[0]:
                best = (score, prediction.label, x_w, z_w)
        if best is None:
            return None

        score, label, x_w, z_w = best
        self.history.append((label, score, x_w, z_w))
        confirmed = self._confirm(label)
        if confirmed is None:
            return None
        x_med, z_med, confidence = confirmed
        return DetectionEvent(x_med, z_med, label, confidence, self.cfg.CNN_CONFIRM_N, timestamp_s)

    def _estimate_world(
        self,
        pose: Pose,
        center_x_px: float,
        box_area_px: float,
        frame_area_px: float,
        frame_width_px: int,
        front_range_m: float,
    ) -> tuple[float, float]:
        bearing = normalize_angle(
            pose.yaw_rad + ((center_x_px / max(1.0, frame_width_px)) - 0.5) * self.cfg.CAMERA_FOV_RAD
        )
        area_ratio = max(1e-5, min(1.0, box_area_px / max(1.0, frame_area_px)))
        area_distance = max(self.cfg.CAMERA_MIN_DISTANCE_M, min(self.cfg.CAMERA_MAX_DISTANCE_M, 0.13 / math.sqrt(area_ratio)))
        if front_range_m < self.cfg.SCAN_MAX_RANGE_M * 0.92:
            distance = min(area_distance, max(self.cfg.CAMERA_MIN_DISTANCE_M, front_range_m + 0.035))
        else:
            distance = area_distance
        x_w = pose.x_w + math.cos(bearing) * distance
        z_w = pose.z_w + math.sin(bearing) * distance
        return (
            max(self.cfg.WORLD_X_MIN, min(self.cfg.WORLD_X_MAX, x_w)),
            max(self.cfg.WORLD_Z_MIN, min(self.cfg.WORLD_Z_MAX, z_w)),
        )

    def _confirm(self, label: str) -> tuple[float, float, float] | None:
        recent = [item for item in self.history if item[0] == label]
        if len(recent) < self.cfg.CNN_CONFIRM_N:
            return None
        recent = recent[-self.cfg.CNN_CONFIRM_N :]
        x_med = statistics.median(item[2] for item in recent)
        z_med = statistics.median(item[3] for item in recent)
        for _label, _score, x_w, z_w in recent:
            if math.hypot(x_w - x_med, z_w - z_med) > self.cfg.DETECTION_SPATIAL_TOLERANCE_M:
                return None
        confidence = sum(item[1] for item in recent) / len(recent)
        return x_med, z_med, confidence
