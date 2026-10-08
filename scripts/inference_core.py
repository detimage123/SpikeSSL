#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import scipy.io as sio
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

from src.models.components.spikessm_mm_v2_var import SpikeSSMMultiModalModel_v2_var
from src.utils.signal_processing import smooth_spikes


DEFAULT_INPUT_ROOT = Path(ROOT) / "data"
DEFAULT_CKPT = Path(ROOT) / "checkpoints" / "spikessl_ui_default.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared utilities for SpikeSSL calcium-trace inference."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--glob", default="**/*.mat")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(ROOT) / "result" / f"spikessl_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", nargs="+", choices=("soma", "dend"), default=("soma", "dend"))
    parser.add_argument("--viz-count", type=int, default=6)
    parser.add_argument("--zoom-frames", type=int, default=3000)
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=1,
        help="Downsample traces on the time axis before inference/plotting; 5 means every 5 frames become 1 frame.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=("mean", "stride"),
        default="mean",
        help="mean uses non-overlapping block average; stride keeps every Nth frame.",
    )
    parser.add_argument("--private-indicator-id", type=int, default=0)
    parser.add_argument(
        "--input-transform",
        choices=("zscore", "detrend_zscore", "dff_zscore", "diff_zscore"),
        default="zscore",
    )
    parser.add_argument("--input-detrend-window-sec", type=float, default=20.0)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--use-pred-postproc", action="store_true")
    parser.add_argument("--pred-postproc-smooth-sec", type=float, default=0.025)
    parser.add_argument("--pred-postproc-threshold-mad", type=float, default=0.75)
    parser.add_argument("--pred-postproc-resmooth-sec", type=float, default=0.025)
    return parser.parse_args()


def build_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load the released inference checkpoint without initializing Lightning.

    The training checkpoint stores its network parameters with a ``net.``
    prefix. The inference UI only needs the network, so loading it directly
    avoids training-only imports and makes startup substantially lighter.
    """
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = SpikeSSMMultiModalModel_v2_var(
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


def scalar_or_none(data: dict, key: str):
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return str(data[key])


def parse_compression_ratio(path: Path, data: dict) -> float | None:
    value = scalar_or_none(data, "compression_ratio")
    if value is not None:
        return float(value)
    text = "/".join(path.parts[-3:])
    match = re.search(r"(?:^|[_/])cr(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0
    return None


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


def save_paired_evalstyle_plot(
    output_path: Path,
    title: str,
    raw_traces: np.ndarray,
    recon_traces: np.ndarray,
    raw_preds: np.ndarray,
    recon_preds: np.ndarray,
    trial_ids: list[int],
    raw_metrics: list[dict],
    recon_metrics: list[dict],
    zoom_frames: int,
) -> None:
    raw_z = Predictor.zscore_2d(raw_traces)
    recon_z = Predictor.zscore_2d(recon_traces)
    raw_metric_map = {m["index"]: m for m in raw_metrics}
    recon_metric_map = {m["index"]: m for m in recon_metrics}

    fig, axes = plt.subplots(
        len(trial_ids) * 2,
        4,
        figsize=(36, max(10, 2.8 * len(trial_ids) * 2)),
        squeeze=False,
    )

    for row, tid in enumerate(trial_ids):
        top = row * 2
        raw_input_ax = axes[top, 0]
        recon_input_ax = axes[top, 1]
        raw_zoom_ax = axes[top, 2]
        recon_zoom_ax = axes[top, 3]
        raw_pred_ax = axes[top + 1, 0]
        recon_pred_ax = axes[top + 1, 1]
        raw_pred_zoom_ax = axes[top + 1, 2]
        recon_pred_zoom_ax = axes[top + 1, 3]

        length = min(raw_traces.shape[1], recon_traces.shape[1], raw_preds.shape[1], recon_preds.shape[1])
        t = np.arange(length)
        zoom_start, zoom_end = pick_zoom_window(raw_preds[tid], recon_preds[tid], zoom_frames)
        tz = np.arange(zoom_start, zoom_end)
        raw_stat = raw_metric_map[tid]
        recon_stat = recon_metric_map[tid]

        raw_input_ax.plot(t, raw_z[tid, :length], color="steelblue", linewidth=0.8, alpha=0.9)
        raw_input_ax.set_title(f"Trial {tid + 1} raw/gt input  trace_std={raw_stat['trace_std']:.3f}", fontsize=9)
        raw_input_ax.set_ylabel("z-score")

        recon_input_ax.plot(t, recon_z[tid, :length], color="steelblue", linewidth=0.8, alpha=0.9)
        recon_input_ax.set_title(f"Trial {tid + 1} recon input  trace_std={recon_stat['trace_std']:.3f}", fontsize=9)

        raw_pred_ax.plot(t, raw_preds[tid, :length], color="navy", linewidth=0.9, alpha=0.95)
        raw_pred_ax.set_title(
            f"Trial {tid + 1} raw/gt pred  mean={raw_stat['pred_mean']:.3f}  max={raw_stat['pred_max']:.3f}",
            fontsize=9,
        )
        raw_pred_ax.set_ylabel("pred")

        recon_pred_ax.plot(t, recon_preds[tid, :length], color="navy", linewidth=0.9, alpha=0.95)
        recon_pred_ax.set_title(
            f"Trial {tid + 1} recon pred  mean={recon_stat['pred_mean']:.3f}  max={recon_stat['pred_max']:.3f}",
            fontsize=9,
        )

        raw_zoom_ax.plot(tz, raw_z[tid, zoom_start:zoom_end], color="steelblue", linewidth=0.8, alpha=0.9)
        raw_zoom_ax.set_title(f"zoom raw/gt input [{zoom_start}:{zoom_end}]", fontsize=9)

        recon_zoom_ax.plot(tz, recon_z[tid, zoom_start:zoom_end], color="steelblue", linewidth=0.8, alpha=0.9)
        recon_zoom_ax.set_title(f"zoom recon input [{zoom_start}:{zoom_end}]", fontsize=9)

        raw_pred_zoom_ax.plot(tz, raw_preds[tid, zoom_start:zoom_end], color="navy", linewidth=0.9, alpha=0.95)
        raw_pred_zoom_ax.set_title(
            f"zoom raw/gt pred [{zoom_start}:{zoom_end}]  mass={float(raw_preds[tid, zoom_start:zoom_end].sum()):.3f}",
            fontsize=9,
        )

        recon_pred_zoom_ax.plot(tz, recon_preds[tid, zoom_start:zoom_end], color="navy", linewidth=0.9, alpha=0.95)
        recon_pred_zoom_ax.set_title(
            f"zoom recon pred [{zoom_start}:{zoom_end}]  mass={float(recon_preds[tid, zoom_start:zoom_end].sum()):.3f}",
            fontsize=9,
        )

        ymax = max(
            1e-3,
            float(np.quantile(raw_preds[tid], 0.995)),
            float(np.quantile(recon_preds[tid], 0.995)),
            float(np.quantile(raw_preds[tid, zoom_start:zoom_end], 0.995)),
            float(np.quantile(recon_preds[tid, zoom_start:zoom_end], 0.995)),
        )
        for ax in (raw_pred_ax, recon_pred_ax, raw_pred_zoom_ax, recon_pred_zoom_ax):
            ax.set_ylim(0.0, ymax * 1.15)
        for ax in axes[top : top + 2].reshape(-1):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("frame")
    fig.suptitle(title, fontsize=14, y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.992))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_mode(
    predictor: Predictor,
    model: torch.nn.Module,
    mat_path: Path,
    data: dict,
    mode: str,
    mode_dir: Path,
    args: argparse.Namespace,
) -> dict:
    raw_traces_original = predictor.ensure_2d(data[f"raw_trace_{mode}"])
    recon_traces_original = predictor.ensure_2d(data[f"recon_trace_{mode}"])
    raw_traces = temporal_downsample(raw_traces_original, args.temporal_downsample, args.downsample_method)
    recon_traces = temporal_downsample(recon_traces_original, args.temporal_downsample, args.downsample_method)
    raw_pred_raw = predictor.predict_reconstructed(model, raw_traces)
    recon_pred_raw = predictor.predict_reconstructed(model, recon_traces)
    raw_preds = predictor.postprocess_pred(raw_pred_raw)
    recon_preds = predictor.postprocess_pred(recon_pred_raw)

    trial_ids, recon_metrics = select_trials(recon_traces, recon_preds, args.viz_count)
    _, raw_metrics = select_trials(raw_traces, raw_preds, args.viz_count)
    title = f"{mat_path.relative_to(args.input_root)} / {mode} paired raw/gt vs recon trace and SpikeSSM pred"
    save_paired_evalstyle_plot(
        output_path=mode_dir / "eval_style_gt_recon_spike_paired.png",
        title=title,
        raw_traces=raw_traces,
        recon_traces=recon_traces,
        raw_preds=raw_preds,
        recon_preds=recon_preds,
        trial_ids=trial_ids,
        raw_metrics=raw_metrics,
        recon_metrics=recon_metrics,
        zoom_frames=args.zoom_frames,
    )
    np.savez_compressed(
        mode_dir / "predictions_paired.npz",
        raw_traces=raw_traces.astype(np.float32),
        recon_traces=recon_traces.astype(np.float32),
        raw_pred_raw=raw_pred_raw.astype(np.float32),
        recon_pred_raw=recon_pred_raw.astype(np.float32),
        raw_preds=raw_preds.astype(np.float32),
        recon_preds=recon_preds.astype(np.float32),
        viz_indices=np.asarray(trial_ids, dtype=np.int64),
    )
    summary = {
        "mode": mode,
        "n_traces": int(raw_traces.shape[0]),
        "original_trace_length": int(raw_traces_original.shape[1]),
        "trace_length": int(raw_traces.shape[1]),
        "temporal_downsample": int(args.temporal_downsample),
        "downsample_method": str(args.downsample_method),
        "viz_indices": trial_ids,
        "raw_pred_stats": pred_stats(raw_preds),
        "recon_pred_stats": pred_stats(recon_preds),
    }
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_mat(
    predictor: Predictor,
    model: torch.nn.Module,
    mat_path: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    rel = mat_path.relative_to(args.input_root)
    item_dir = output_root / rel.with_suffix("")
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    compression_ratio = parse_compression_ratio(mat_path, data)
    qp = scalar_or_none(data, "qp")

    print(f"[MAT Start] {rel} cr={compression_ratio} qp={qp}", flush=True)
    manifest = {
        "mat_path": str(mat_path),
        "rel_path": str(rel),
        "compression_ratio": compression_ratio,
        "qp": qp,
        "ckpt": str(args.ckpt),
        "input_transform": args.input_transform,
        "temporal_downsample": args.temporal_downsample,
        "downsample_method": args.downsample_method,
        "private_indicator_id": args.private_indicator_id,
        "zoom_frames": args.zoom_frames,
        "viz_count": args.viz_count,
        "selection": "same trial index, selected by recon prediction mass using previous eval-style rule",
    }
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {}
    for mode in list(dict.fromkeys(args.modes)):
        mode_dir = item_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        result[mode] = run_mode(predictor, model, mat_path, data, mode, mode_dir, args)
        print(
            f"[Mode Done] {rel} mode={mode} raw_mean={result[mode]['raw_pred_stats']['mean']:.3f} "
            f"recon_mean={result[mode]['recon_pred_stats']['mean']:.3f}",
            flush=True,
        )
    (item_dir / "summary_all.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, dm_hp = build_model(args.ckpt.resolve(), device)
    predictor = Predictor(args, dm_hp)
    files = sorted([p for p in args.input_root.glob(args.glob) if p.is_file()])
    if args.max_files > 0:
        files = files[: args.max_files]
    all_summary = {}
    for mat_path in files:
        all_summary[str(mat_path.relative_to(args.input_root))] = run_mat(predictor, model, mat_path, args.output_dir, args)
    (args.output_dir / "summary_all.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[All Done] files={len(files)} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
