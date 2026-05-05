"""Terrain classification from world bounds and floor color."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np


@dataclass(frozen=True)
class TerrainZone:
    kind: str
    name: str
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    penalty_cost: float = 0.0

    def contains(self, x_w: float, z_w: float) -> bool:
        return self.x_min <= x_w <= self.x_max and self.z_min <= z_w <= self.z_max


@dataclass(frozen=True)
class TerrainSample:
    kind: str
    name: str
    penalty_cost: float = 0.0
    source: str = "zone"


@dataclass(frozen=True)
class TerrainColorPrototype:
    kind: str
    name: str
    rgb: tuple[float, float, float]
    penalty_cost: float


class TerrainClassifier:
    _priority = {"trap": 0, "hazard": 1, "swamp": 2, "checkpoint": 3, "start": 4}

    def __init__(
        self,
        zone_specs=(),
        color_specs=(),
        color_match_max_distance: float = 0.22,
        color_center_crop_ratio: float = 0.75,
        dark_max_brightness: float = 0.075,
        trap_max_channel: float = 0.075,
        trap_max_chroma: float = 0.035,
    ) -> None:
        self.zones = [
            TerrainZone(str(kind), str(name), float(x0), float(x1), float(z0), float(z1), float(cost))
            for kind, name, x0, x1, z0, z1, cost in zone_specs
        ]
        self.color_prototypes = [
            TerrainColorPrototype(str(kind), str(name), tuple(float(v) for v in rgb), float(cost))
            for kind, name, rgb, cost in color_specs
        ]
        self.color_match_max_distance = float(color_match_max_distance)
        self.color_center_crop_ratio = float(color_center_crop_ratio)
        self.dark_max_brightness = float(dark_max_brightness)
        self.trap_max_channel = float(trap_max_channel)
        self.trap_max_chroma = float(trap_max_chroma)

    def sample(self, x_w: float, z_w: float, floor_rgb: np.ndarray | None = None) -> TerrainSample | None:
        zone = self._sample_zone(x_w, z_w)
        color = self.sample_from_rgb(floor_rgb)
        if color is not None and color.kind in ("trap", "hazard"):
            return color
        return color or zone

    def _sample_zone(self, x_w: float, z_w: float) -> TerrainSample | None:
        matches = [zone for zone in self.zones if zone.contains(x_w, z_w)]
        if not matches:
            return None
        best = min(matches, key=lambda zone: self._priority.get(zone.kind, 99))
        return TerrainSample(best.kind, best.name, best.penalty_cost, "zone")

    def sample_from_rgb(self, floor_rgb: np.ndarray | None) -> TerrainSample | None:
        if floor_rgb is None or not self.color_prototypes:
            return None
        frame = np.asarray(floor_rgb, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        if frame.max() > 1.0:
            frame = frame / 255.0

        h, w = frame.shape[:2]
        crop_ratio = min(1.0, max(0.2, self.color_center_crop_ratio))
        ch = max(1, int(round(h * crop_ratio)))
        cw = max(1, int(round(w * crop_ratio)))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        mean_rgb = frame[y0 : y0 + ch, x0 : x0 + cw, :3].mean(axis=(0, 1))
        brightness = float(mean_rgb.mean())
        chroma = float(mean_rgb.max() - mean_rgb.min())

        if brightness <= self.dark_max_brightness and mean_rgb.max() <= self.trap_max_channel and chroma <= self.trap_max_chroma:
            return TerrainSample("trap", "hole_tile", 0.0, "color")
        if brightness > 0.88 and chroma < 0.08:
            return None

        best: tuple[float, TerrainColorPrototype] | None = None
        for prototype in self.color_prototypes:
            if prototype.kind == "trap":
                continue
            dist = math.sqrt(sum((float(mean_rgb[idx]) - prototype.rgb[idx]) ** 2 for idx in range(3)))
            if best is None or dist < best[0]:
                best = (dist, prototype)
        if best is None or best[0] > self.color_match_max_distance:
            return None
        proto = best[1]
        return TerrainSample(proto.kind, proto.name, proto.penalty_cost, "color")


def parse_world_terrain_zones(
    world_path: str | Path,
    swamp_penalty: float,
    hazard_margin_m: float = 0.0,
) -> list[tuple[str, str, float, float, float, float, float]]:
    path = Path(world_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"DEF\s+(checkpoint|trap|swamp)(\d+)(min|max)\s+Transform\s*\{[^{}]*?"
        r"translation\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
        r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        re.DOTALL,
    )
    pairs: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for match in pattern.finditer(text):
        kind, idx, endpoint, x_raw, z_raw = match.groups()
        pairs.setdefault((kind, idx), {})[endpoint] = (float(x_raw), float(z_raw))

    zones: list[tuple[str, str, float, float, float, float, float]] = []
    for (kind, idx), endpoints in sorted(pairs.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        if "min" not in endpoints or "max" not in endpoints:
            continue
        x0, z0 = endpoints["min"]
        x1, z1 = endpoints["max"]
        margin = hazard_margin_m if kind == "trap" else 0.0
        x_min = min(x0, x1) - margin
        x_max = max(x0, x1) + margin
        z_min = min(z0, z1) - margin
        z_max = max(z0, z1) + margin
        cost = float(swamp_penalty) if kind == "swamp" else 0.0
        zones.append((kind, f"{kind}{idx}_bounds", x_min, x_max, z_min, z_max, cost))
    return zones
