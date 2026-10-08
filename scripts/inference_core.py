#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyrootutils
import torch

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path = [p for p in sys.path if ".local/lib/python" not in p]

from src.models.components.spikessl import SpikeSSL
from src.utils.signal_processing import smooth_spikes


DEFAULT_CKPT = Path(ROOT) / "checkpoints" / "spikessl_ui_default.ckpt"


def build_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load the released inference checkpoint without initializing Lightning.

    The training checkpoint stores its network parameters with a ``net.``
    prefix. The inference UI only needs the network, so loading it directly
    avoids training-only imports and makes startup substantially lighter.
    """
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = SpikeSSL(
        in_features=1,
        d_model=192,
        d_cond=64,
        n_blocks=8,
        d_conv=7,
        expand=2,
        refine_layers=3,
        dropout=0.1,
        use_sigmoid=False,
        use_gate=True,
        gate_threshold=0.5,
        use_checkpoint=True,
        var_warmup_epochs=2,
        fewshot_mode=False,
    )
    state_dict = {
        key.removeprefix("net."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("net.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    unexpected_missing = [key for key in missing if key != "target_prompt"]
    if unexpected_missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {unexpected_missing}")
    model.to(device)
    model.eval()
    return model, checkpoint.get("datamodule_hyper_parameters", {})


class Predictor:
    def __init__(self, args: argparse.Namespace, dm_hp: dict) -> None:
        self.frame_rate = float(args.frame_rate)
        self.segment_len = int(dm_hp.get("segment_len", 128))
        self.context_left = int(dm_hp.get("context_left", 30))
        self.context_right = int(dm_hp.get("context_right", 30))
        self.overlap_size = int(dm_hp.get("overlap_size", 4))
        self.infer_batch_size = int(args.infer_batch_size)
        self.private_indicator_id = int(args.private_indicator_id)
        self.segment_total_len = self.segment_len + self.context_left + self.context_right
        self.stride = max(1, self.segment_total_len // max(1, self.overlap_size))
        self.input_transform = str(args.input_transform)
        self.input_detrend_window_sec = float(args.input_detrend_window_sec)
        self.use_pred_postproc = bool(args.use_pred_postproc)
        self.pred_postproc_smooth_sec = float(args.pred_postproc_smooth_sec)
        self.pred_postproc_threshold_mad = float(args.pred_postproc_threshold_mad)
        self.pred_postproc_resmooth_sec = float(args.pred_postproc_resmooth_sec)

    @staticmethod
    def ensure_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D trace array, got {arr.shape}")
        return arr

    @staticmethod
    def zscore_2d(arr: np.ndarray) -> np.ndarray:
        arr = Predictor.ensure_2d(arr)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        return np.where(std > 1e-8, (arr - mean) / (std + 1e-8), arr - mean)

    @staticmethod
    def moving_average_2d(arr: np.ndarray, window: int) -> np.ndarray:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        out = np.empty_like(arr, dtype=np.float32)
        left = window // 2
        right = window - 1 - left
        for i, trace in enumerate(np.asarray(arr, dtype=np.float32, copy=False)):
            padded = np.pad(trace, (left, right), mode="edge")
            out[i] = np.convolve(padded, kernel, mode="valid")
        return out

    def smooth_rate_2d(self, arr: np.ndarray, smoothing_sec: float) -> np.ndarray:
        sigma_frames = max(1.0, float(smoothing_sec) * self.frame_rate)
        out = np.empty_like(arr, dtype=np.float32)
        for i, trace in enumerate(np.asarray(arr, dtype=np.float32, copy=False)):
            out[i] = smooth_spikes(trace.astype(np.float64), smoothing=sigma_frames, causal_kernel=False)
        return out

    def postprocess_pred(self, pred: np.ndarray) -> np.ndarray:
        arr = np.asarray(pred, dtype=np.float32)
        if not self.use_pred_postproc:
            return arr
        smoothed = self.smooth_rate_2d(arr, self.pred_postproc_smooth_sec)
        med = np.median(smoothed, axis=1, keepdims=True)
        mad = np.median(np.abs(smoothed - med), axis=1, keepdims=True)
        robust_std = 1.4826 * np.maximum(mad, 1e-6)
        sparse = np.clip(smoothed - (med + self.pred_postproc_threshold_mad * robust_std), 0.0, None)
        if self.pred_postproc_resmooth_sec > 0:
            sparse = self.smooth_rate_2d(sparse, self.pred_postproc_resmooth_sec)
        return sparse.astype(np.float32, copy=False)

    def transform_input_traces(self, traces: np.ndarray) -> np.ndarray:
        arr = self.ensure_2d(traces)
        if self.input_transform == "zscore":
            return self.zscore_2d(arr).astype(np.float32, copy=False)
        if self.input_transform == "detrend_zscore":
            window = max(3, int(round(self.input_detrend_window_sec * self.frame_rate)))
            if window % 2 == 0:
                window += 1
            return self.zscore_2d(arr - self.moving_average_2d(arr, window)).astype(np.float32, copy=False)
        if self.input_transform == "dff_zscore":
            window = max(3, int(round(self.input_detrend_window_sec * self.frame_rate)))
            if window % 2 == 0:
                window += 1
            baseline = self.moving_average_2d(arr, window)
            denom = np.maximum(np.abs(baseline), np.percentile(np.abs(arr), 5, axis=1, keepdims=True) + 1e-6)
            return self.zscore_2d((arr - baseline) / denom).astype(np.float32, copy=False)
        if self.input_transform == "diff_zscore":
            return self.zscore_2d(np.diff(arr, axis=1, prepend=arr[:, :1])).astype(np.float32, copy=False)
        raise ValueError(f"Unknown input_transform: {self.input_transform}")

    def forward_model(self, model: torch.nn.Module, traces: np.ndarray) -> np.ndarray:
        device = next(model.parameters()).device
        x = torch.from_numpy(np.asarray(traces, dtype=np.float32)).unsqueeze(-1).to(device)
        indicator_id = torch.full((x.shape[0],), self.private_indicator_id, dtype=torch.long, device=device)
        fs = torch.full((x.shape[0],), self.frame_rate, dtype=torch.float32, device=device)
        with torch.inference_mode():
            pred = model(x, indicator_id=indicator_id, fs=fs)
        return pred[..., 0].detach().cpu().numpy().clip(min=0.0)

    def window_starts(self, trial_len: int) -> list[int]:
        if trial_len <= self.segment_total_len:
            return [0]
        starts = np.arange(0, trial_len - self.segment_total_len + 1, step=self.stride).tolist()
        last_possible_start = trial_len - self.segment_total_len
        if not starts or starts[-1] != last_possible_start:
            starts.append(last_possible_start)
        return starts

    def predict_reconstructed(self, model: torch.nn.Module, traces: np.ndarray) -> np.ndarray:
        traces_z = self.transform_input_traces(traces)
        n_traces, trial_len = traces_z.shape
        pred_sum = np.zeros((n_traces, trial_len), dtype=np.float32)
        pred_count = np.zeros((n_traces, trial_len), dtype=np.float32)
        starts = self.window_starts(trial_len)
        valid_slice_end = -self.context_right if self.context_right > 0 else None

        batch_windows: list[np.ndarray] = []
        batch_meta: list[tuple[int, int, int]] = []

        def flush_batch() -> None:
            nonlocal batch_windows, batch_meta
            if not batch_windows:
                return
            pred_batch = self.forward_model(model, np.stack(batch_windows, axis=0))
            for i, (trace_idx, valid_start, valid_end) in enumerate(batch_meta):
                valid_pred = pred_batch[i, self.context_left:valid_slice_end]
                pred_sum[trace_idx, valid_start:valid_end] += valid_pred[: valid_end - valid_start]
                pred_count[trace_idx, valid_start:valid_end] += 1.0
            batch_windows = []
            batch_meta = []

        for trace_idx in range(n_traces):
            for start in starts:
                end = min(start + self.segment_total_len, trial_len)
                window = traces_z[trace_idx, start:end]
                if len(window) < self.segment_total_len:
                    window = np.pad(window, (0, self.segment_total_len - len(window)), mode="edge")
                valid_start = start + self.context_left
                valid_end = min(start + self.segment_total_len - self.context_right, trial_len)
                if valid_start >= valid_end:
                    continue
                batch_windows.append(window.astype(np.float32, copy=False))
                batch_meta.append((trace_idx, valid_start, valid_end))
                if len(batch_windows) >= self.infer_batch_size:
                    flush_batch()
        flush_batch()
        pred_count[pred_count == 0] = 1.0
        return (pred_sum / pred_count).astype(np.float32, copy=False)


def pred_stats(pred: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(pred)),
        "std": float(np.std(pred)),
        "max": float(np.max(pred)),
        "active_frac_0p10": float(np.mean(pred > 0.10)),
        "active_frac_0p20": float(np.mean(pred > 0.20)),
    }


def pick_zoom_window(pred_a: np.ndarray, pred_b: np.ndarray, zoom_frames: int) -> tuple[int, int]:
    length = min(len(pred_a), len(pred_b))
    if length <= zoom_frames:
        return 0, length
    score = np.maximum(pred_a[:length], pred_b[:length]).astype(np.float64)
    window_score = np.convolve(score, np.ones(zoom_frames, dtype=np.float64), mode="valid")
    start = int(np.argmax(window_score))
    return start, min(start + zoom_frames, length)


def select_trials(traces: np.ndarray, preds: np.ndarray, viz_count: int) -> tuple[list[int], list[dict]]:
    metrics = []
    for i in range(preds.shape[0]):
        metrics.append(
            {
                "index": int(i),
                "trace_std": float(np.std(traces[i])),
                "pred_mean": float(np.mean(preds[i])),
                "pred_max": float(np.max(preds[i])),
                "pred_mass": float(np.sum(preds[i])),
            }
        )
    head_ids = list(range(min(3, len(metrics))))
    remaining = [m for m in metrics if m["index"] not in set(head_ids)]
    tail_ids = [m["index"] for m in sorted(remaining, key=lambda x: x["pred_mass"], reverse=True)]
    selected = (head_ids + tail_ids)[:viz_count]
    if len(selected) < viz_count:
        for m in metrics:
            if m["index"] not in selected:
                selected.append(m["index"])
            if len(selected) >= viz_count:
                break
    return selected[:viz_count], metrics


def temporal_downsample(traces: np.ndarray, factor: int, method: str) -> np.ndarray:
    arr = Predictor.ensure_2d(traces)
    if factor <= 1:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] < factor:
        raise ValueError(f"Trace length {arr.shape[1]} is shorter than downsample factor {factor}")
    if method == "stride":
        return arr[:, ::factor].astype(np.float32, copy=False)
    if method == "mean":
        usable_len = (arr.shape[1] // factor) * factor
        return arr[:, :usable_len].reshape(arr.shape[0], -1, factor).mean(axis=2).astype(np.float32, copy=False)
    raise ValueError(f"Unknown downsample method: {method}")
