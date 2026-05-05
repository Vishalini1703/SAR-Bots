"""Classical CV candidate generation for wall token regions."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CandidateBox:
    x: int
    y: int
    w: int
    h: int
    score: float


class CVCandidateGenerator:
    def __init__(
        self,
        min_area_px: int,
        max_area_px: int,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 4.0,
    ) -> None:
        self.min_area_px = int(min_area_px)
        self.max_area_px = int(max_area_px)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

    def generate(self, rgb_frame: np.ndarray) -> list[CandidateBox]:
        if rgb_frame is None or rgb_frame.size == 0:
            return []
        frame = np.asarray(rgb_frame, dtype=np.float32)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        mean = float(gray.mean())
        std = max(12.0, float(gray.std()))

        bright = gray > min(245.0, mean + 1.15 * std)
        dark = gray < max(8.0, mean - 1.10 * std)
        chroma = frame.max(axis=2) - frame.min(axis=2)
        colored_panel = (frame[:, :, 0] > 110.0) & (frame[:, :, 1] > 90.0) & (chroma > 25.0)
        mask = bright | dark | colored_panel
        mask = self._morph_close(mask, 3)
        boxes = self._components(mask)
        if boxes:
            return boxes[:8]
        return self._fallback_windows(rgb_frame.shape[1], rgb_frame.shape[0])

    def _components(self, mask: np.ndarray) -> list[CandidateBox]:
        height, width = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        boxes: list[CandidateBox] = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                pixels = self._flood(mask, visited, x, y)
                if not pixels:
                    continue
                xs = [p[0] for p in pixels]
                ys = [p[1] for p in pixels]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                box_w = max_x - min_x + 1
                box_h = max_y - min_y + 1
                area = box_w * box_h
                if area < self.min_area_px or area > self.max_area_px:
                    continue
                aspect = box_w / max(1.0, float(box_h))
                if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                    continue
                fill = len(pixels) / float(area)
                pad_x = max(2, int(round(box_w * 0.18)))
                pad_y = max(2, int(round(box_h * 0.18)))
                x0 = max(0, min_x - pad_x)
                y0 = max(0, min_y - pad_y)
                x1 = min(width, max_x + pad_x + 1)
                y1 = min(height, max_y + pad_y + 1)
                score = min(1.0, 0.45 + 0.55 * fill)
                boxes.append(CandidateBox(x0, y0, x1 - x0, y1 - y0, score))
        boxes.sort(key=lambda box: box.score * box.w * box.h, reverse=True)
        return boxes

    def _flood(self, mask: np.ndarray, visited: np.ndarray, start_x: int, start_y: int) -> list[tuple[int, int]]:
        width = mask.shape[1]
        height = mask.shape[0]
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            pixels.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))
        return pixels

    def _morph_close(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        return self._erode(self._dilate(mask, kernel_size), kernel_size)

    @staticmethod
    def _dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
        pad = kernel_size // 2
        padded = np.pad(mask, pad, mode="constant", constant_values=False)
        result = np.zeros(mask.shape, dtype=bool)
        for y in range(kernel_size):
            for x in range(kernel_size):
                result |= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
        return result

    @staticmethod
    def _erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
        pad = kernel_size // 2
        padded = np.pad(mask, pad, mode="constant", constant_values=False)
        result = np.ones(mask.shape, dtype=bool)
        for y in range(kernel_size):
            for x in range(kernel_size):
                result &= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
        return result

    def _fallback_windows(self, width: int, height: int) -> list[CandidateBox]:
        box_w = max(12, width // 3)
        box_h = max(12, height // 2)
        y = max(0, (height - box_h) // 2)
        xs = [max(0, (width - box_w) // 2), max(0, width // 5 - box_w // 2), min(width - box_w, 4 * width // 5 - box_w // 2)]
        return [CandidateBox(x, y, box_w, box_h, 0.25) for x in xs]
