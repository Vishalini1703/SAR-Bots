"""Lightweight NumPy CNN for H/S/U wall-token classification."""

from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class TinyCNNClassifier:
    def __init__(self, input_size: int, labels: tuple[str, ...], weights_path: str | None = None) -> None:
        self.input_size = int(input_size)
        self.labels = tuple(labels)
        self.conv1_w: np.ndarray | None = None
        self.conv1_b: np.ndarray | None = None
        self.conv2_w: np.ndarray | None = None
        self.conv2_b: np.ndarray | None = None
        self.fc_w: np.ndarray | None = None
        self.fc_b: np.ndarray | None = None
        self.pool_size = 2
        if weights_path:
            self._load(weights_path)

    def _load(self, weights_path: str) -> None:
        resolved = self._resolve(weights_path)
        if resolved is None:
            print(f"event cnn_weights_missing path={weights_path}")
            return
        data = np.load(resolved)
        if "labels" in data:
            self.labels = tuple(str(v) for v in data["labels"].tolist())
        self.input_size = int(data["input_size"]) if "input_size" in data else self.input_size
        self.pool_size = int(data["pool_size"]) if "pool_size" in data else self.pool_size
        self.conv1_w = np.asarray(data["conv1_w"], dtype=np.float32)
        self.conv1_b = np.asarray(data["conv1_b"], dtype=np.float32)
        self.conv2_w = np.asarray(data["conv2_w"], dtype=np.float32)
        self.conv2_b = np.asarray(data["conv2_b"], dtype=np.float32)
        self.fc_w = np.asarray(data["fc_w"], dtype=np.float32)
        self.fc_b = np.asarray(data["fc_b"], dtype=np.float32)
        print(f"event cnn_weights_loaded path={resolved} labels={self.labels}")

    def _resolve(self, weights_path: str) -> str | None:
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            weights_path,
            os.path.join(base, weights_path),
            os.path.join(base, os.path.basename(weights_path)),
            os.path.join(base, "..", "..", weights_path),
        ]
        for candidate in candidates:
            resolved = os.path.abspath(candidate)
            if os.path.exists(resolved):
                return resolved
        return None

    def predict(self, rgb_crop: np.ndarray) -> Prediction:
        x = self._preprocess(rgb_crop)
        if x is None or self.conv1_w is None or self.fc_w is None:
            return Prediction(self.labels[0] if self.labels else "H", 0.0)
        x = x.reshape(1, self.input_size, self.input_size)
        x = self._relu(self._conv2d(x, self.conv1_w, self.conv1_b))
        x = self._max_pool(x, 2, 2)
        x = self._relu(self._conv2d(x, self.conv2_w, self.conv2_b))
        x = self._max_pool(x, 2, 2)
        x = self._adaptive_avg_pool(x, self.pool_size, self.pool_size).reshape(-1)
        logits = self.fc_w @ x + self.fc_b
        probs = self._softmax(logits)
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else "H"
        return Prediction(label, float(probs[idx]))

    def _preprocess(self, rgb_crop: np.ndarray) -> np.ndarray | None:
        if rgb_crop is None or rgb_crop.size == 0:
            return None
        frame = np.asarray(rgb_crop, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        if frame.max() > 1.0:
            frame = frame / 255.0
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        lo = float(np.percentile(gray, 5))
        hi = float(np.percentile(gray, 95))
        if hi - lo > 1e-4:
            gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
        mask = gray < 0.72
        if np.any(mask):
            ys, xs = np.where(mask)
            pad_y = max(1, int(round((ys.max() - ys.min() + 1) * 0.25)))
            pad_x = max(1, int(round((xs.max() - xs.min() + 1) * 0.25)))
            y0 = max(0, int(ys.min()) - pad_y)
            y1 = min(gray.shape[0], int(ys.max()) + pad_y + 1)
            x0 = max(0, int(xs.min()) - pad_x)
            x1 = min(gray.shape[1], int(xs.max()) + pad_x + 1)
            gray = gray[y0:y1, x0:x1]
        resized = self._resize_nn(gray, self.input_size, self.input_size)
        return None if resized is None else resized.astype(np.float32)

    @staticmethod
    def _resize_nn(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray | None:
        in_h, in_w = img.shape[:2]
        if in_h <= 0 or in_w <= 0:
            return None
        y_idx = np.clip((np.arange(out_h) * in_h / out_h).astype(int), 0, in_h - 1)
        x_idx = np.clip((np.arange(out_w) * in_w / out_w).astype(int), 0, in_w - 1)
        return img[y_idx[:, None], x_idx[None, :]]

    @staticmethod
    def _conv2d(x: np.ndarray, kernels: np.ndarray, bias: np.ndarray) -> np.ndarray:
        channels, height, width = x.shape
        out_channels = kernels.shape[0]
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        out = np.zeros((out_channels, height, width), dtype=np.float32)
        for oc in range(out_channels):
            acc = np.full((height, width), float(bias[oc]), dtype=np.float32)
            for ic in range(channels):
                kernel = kernels[oc, ic]
                for ky in range(3):
                    for kx in range(3):
                        acc += kernel[ky, kx] * padded[ic, ky : ky + height, kx : kx + width]
            out[oc] = acc
        return out

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _max_pool(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
        channels, height, width = x.shape
        out_h = max(1, 1 + (height - kernel) // stride)
        out_w = max(1, 1 + (width - kernel) // stride)
        out = np.zeros((channels, out_h, out_w), dtype=np.float32)
        for y in range(out_h):
            for x_idx in range(out_w):
                patch = x[:, y * stride : y * stride + kernel, x_idx * stride : x_idx * stride + kernel]
                out[:, y, x_idx] = patch.max(axis=(1, 2))
        return out

    @staticmethod
    def _adaptive_avg_pool(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        channels, height, width = x.shape
        result = np.zeros((channels, out_h, out_w), dtype=np.float32)
        for oy in range(out_h):
            y0 = int(np.floor(oy * height / out_h))
            y1 = max(y0 + 1, int(np.ceil((oy + 1) * height / out_h)))
            for ox in range(out_w):
                x0 = int(np.floor(ox * width / out_w))
                x1 = max(x0 + 1, int(np.ceil((ox + 1) * width / out_w)))
                result[:, oy, ox] = x[:, y0:y1, x0:x1].mean(axis=(1, 2))
        return result

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        exp = np.exp(shifted)
        total = float(np.sum(exp))
        return exp / total if total > 0.0 else np.ones_like(exp) / len(exp)
