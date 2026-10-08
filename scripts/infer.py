#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from inference_core import (
    DEFAULT_CKPT,
    ROOT,
    Predictor,
    build_model,
    pick_zoom_window,
    pred_stats,
    select_trials,
    temporal_downsample,
)


DEFAULT_EXAMPLES: list[Path] = []

INDICATOR_NAMES = {
    0: "Unknown Indicator",
    1: "OGB1",
    2: "Cal520",
    3: "GCaMP5k",
    4: "GCaMP6f",
    5: "GCaMP6s",
    6: "GCaMP7f",
    7: "GCaMP8f",
    8: "GCaMP8m",
    9: "GCaMP8s",
    10: "R-CaMP",
    11: "jRCaMP1a",
    12: "jGECO1a",
    13: "XCaMPgf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpikeSSL on fluorescence-trace files.")
    parser.add_argument("--input-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "spikessl_inference")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indicator-id", type=int, default=0)
    parser.add_argument("--temporal-downsample", type=int, default=1)
    parser.add_argument("--downsample-method", choices=("mean", "stride"), default="mean")
    parser.add_argument("--input-transform", choices=("zscore", "detrend_zscore", "dff_zscore", "diff_zscore"), default="zscore")
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--viz-count", type=int, default=6)
    parser.add_argument("--zoom-frames", type=int, default=1000)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--max-trials-per-file", type=int, default=12)
    parser.add_argument("--max-frames-per-trace", type=int, default=0)
    parser.add_argument(
        "--trace-key",
        default="",
        help="For private MAT files, choose one trace field such as raw_trace_soma or recon_trace_dend.",
    )
    parser.add_argument("--source-label", default="", help="Human-readable source label for plot titles.")
    return parser.parse_args()


def normalize_trials(raw) -> list:
    arr = np.asarray(raw, dtype=object)
    if arr.shape == ():
        return [raw]
    return list(arr.reshape(-1))


def load_trial(trial_obj) -> tuple[np.ndarray, np.ndarray, float] | None:
    if hasattr(trial_obj, "fluo_mean") and hasattr(trial_obj, "fluo_time"):
        trace = np.squeeze(trial_obj.fluo_mean).astype(np.float32)
        times = np.squeeze(trial_obj.fluo_time).astype(np.float64)
    else:
        return None
    times = times[np.isfinite(times)]
    trace = trace[: len(times)]
    if len(trace) < 32 or len(times) < 32:
        return None
    frame_rate = float(1.0 / np.mean(np.diff(times)))
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        frame_rate = 30.0
    events = np.asarray(getattr(trial_obj, "events_AP", []), dtype=np.float64).reshape(-1)
    events = events[np.isfinite(events)]
    return trace.astype(np.float32), events, frame_rate


def load_public_mat(path: Path, max_trials: int) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "CAttached" not in data:
        raise ValueError(f"{path} has no CAttached field")
    traces = []
    events = []
    frame_rates = []
    for trial_obj in normalize_trials(data["CAttached"]):
        item = load_trial(trial_obj)
        if item is None:
            continue
        trace, event, frame_rate = item
        traces.append(trace)
        events.append(event)
        frame_rates.append(frame_rate)
        if len(traces) >= max_trials:
            break
    if not traces:
        raise ValueError(f"No usable CAttached trials found: {path}")
    min_len = min(len(trace) for trace in traces)
    traces_2d = np.stack([trace[:min_len] for trace in traces], axis=0).astype(np.float32)
    return traces_2d, events, frame_rates


def load_trace_mat(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, list[np.ndarray], list[float], str]:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "CAttached" in data:
        traces, events, frame_rates = load_public_mat(path, args.max_trials_per_file)
        return traces, events, frame_rates, ""

    trace_key = str(args.trace_key or "raw_trace_soma")
    if trace_key not in data:
        candidates = [k for k in ("raw_trace_soma", "recon_trace_soma", "raw_trace_dend", "recon_trace_dend") if k in data]
        if not candidates:
            raise ValueError(f"{path} has no CAttached field and no private trace fields")
        trace_key = candidates[0]
    traces = np.asarray(data[trace_key], dtype=np.float32)
    if traces.ndim == 1:
        traces = traces[None, :]
    if traces.ndim != 2:
        raise ValueError(f"{path}:{trace_key} should be 1D or 2D, got {traces.shape}")
    traces = traces[: max(1, int(args.max_trials_per_file))]
    frame_rates = [float(args.frame_rate)] * traces.shape[0]
    events = [np.asarray([], dtype=np.float64) for _ in range(traces.shape[0])]
    return traces.astype(np.float32, copy=False), events, frame_rates, trace_key


def make_predictor(args: argparse.Namespace, dm_hp: dict, frame_rate: float) -> Predictor:
    p_args = SimpleNamespace(
        frame_rate=frame_rate,
        infer_batch_size=args.infer_batch_size,
        private_indicator_id=args.indicator_id,
        input_transform=args.input_transform,
        input_detrend_window_sec=20.0,
        use_pred_postproc=False,
        pred_postproc_smooth_sec=0.025,
        pred_postproc_threshold_mad=0.75,
        pred_postproc_resmooth_sec=0.025,
    )
    return Predictor(p_args, dm_hp)


def save_public_plot(
    output_path: Path,
    title: str,
    traces: np.ndarray,
    preds: np.ndarray,
    trial_ids: list[int],
    metrics: list[dict],
    zoom_frames: int,
    temporal_downsample_factor: int,
) -> None:
    trace_z = Predictor.zscore_2d(traces)
    metric_map = {m["index"]: m for m in metrics}
    fig, axes = plt.subplots(
        len(trial_ids) * 2,
        2,
        figsize=(20, max(8, 3.2 * len(trial_ids) * 2)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, tid in enumerate(trial_ids):
        top = row * 2
        trace_ax, trace_zoom_ax = axes[top]
        pred_ax, pred_zoom_ax = axes[top + 1]
        length = min(traces.shape[1], preds.shape[1])
        t = np.arange(length)
        zoom_start, zoom_end = pick_zoom_window(preds[tid], preds[tid], zoom_frames)
        tz = np.arange(zoom_start, zoom_end)
        stat = metric_map[tid]

        trace_ax.plot(t, trace_z[tid, :length], color="steelblue", linewidth=0.8)
        trace_ax.set_title(f"Trial {tid + 1} input trace  trace_std={stat['trace_std']:.3f}", fontsize=9)
        trace_ax.set_ylabel("z-score")
        pred_ax.plot(t, preds[tid, :length], color="navy", linewidth=0.9)
        pred_ax.set_title(f"Trial {tid + 1} pred  mean={stat['pred_mean']:.3f} max={stat['pred_max']:.3f}", fontsize=9)
        pred_ax.set_ylabel("pred")

        for zoom_ax in (trace_zoom_ax, pred_zoom_ax):
            zoom_ax.set_facecolor("#fff7ed")
        processed_frames = zoom_end - zoom_start
        raw_frames = processed_frames * max(1, int(temporal_downsample_factor))
        trace_zoom_ax.plot(tz, trace_z[tid, zoom_start:zoom_end], color="steelblue", linewidth=0.8)
        trace_zoom_ax.set_title(
            f"ZOOM input [{zoom_start}:{zoom_end}]  processed={processed_frames} raw~={raw_frames}",
            fontsize=9,
        )
        pred_zoom_ax.plot(tz, preds[tid, zoom_start:zoom_end], color="navy", linewidth=0.9)
        pred_zoom_ax.set_title(
            f"ZOOM pred [{zoom_start}:{zoom_end}]  processed={processed_frames} raw~={raw_frames}",
            fontsize=9,
        )

        ymax = max(1e-3, float(np.quantile(preds[tid], 0.995)), float(np.quantile(preds[tid, zoom_start:zoom_end], 0.995)))
        for ax in (pred_ax, pred_zoom_ax):
            ax.set_ylim(0.0, ymax * 1.15)
        for ax in axes[top : top + 2].reshape(-1):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("frame")
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_file(args: argparse.Namespace, model, dm_hp: dict, mat_path: Path) -> dict:
    traces, events, frame_rates, used_trace_key = load_trace_mat(mat_path, args)
    frame_rate = float(np.median(frame_rates)) if frame_rates else float(args.frame_rate)
    traces = temporal_downsample(traces, args.temporal_downsample, args.downsample_method)
    if args.max_frames_per_trace > 0 and traces.shape[1] > args.max_frames_per_trace:
        traces = traces[:, : args.max_frames_per_trace]
    predictor = make_predictor(args, dm_hp, frame_rate)
    preds_raw = predictor.predict_reconstructed(model, traces)
    preds = predictor.postprocess_pred(preds_raw)
    trial_ids, metrics = select_trials(traces, preds, args.viz_count)

    item_dir = args.output_dir / mat_path.parent.name / mat_path.stem
    if used_trace_key:
        item_dir = item_dir / used_trace_key
    indicator_name = INDICATOR_NAMES.get(int(args.indicator_id), "Unknown Indicator")
    if args.source_label:
        title_source = args.source_label
    else:
        try:
            mat_path.relative_to(ROOT / "data")
            title_source = "Public Example"
        except ValueError:
            title_source = f"Input {mat_path.stem}"
    if used_trace_key:
        trace_label = used_trace_key.replace("_", " ").title()
        title_source = f"{title_source} - {trace_label}"
    title = f"{title_source} - {indicator_name} - Time-scale {int(args.temporal_downsample)}x"
    traces_per_image = 3
    chunks = [trial_ids[i : i + traces_per_image] for i in range(0, len(trial_ids), traces_per_image)]
    for chunk_idx, chunk_trial_ids in enumerate(chunks, start=1):
        if len(chunks) == 1:
            fig_name = "eval_style_public_spike.png"
            fig_title = title
        else:
            fig_name = f"eval_style_public_spike_part{chunk_idx:02d}.png"
            fig_title = f"{title} - Part {chunk_idx}/{len(chunks)}"
        save_public_plot(
            item_dir / fig_name,
            fig_title,
            traces,
            preds,
            chunk_trial_ids,
            metrics,
            args.zoom_frames,
            args.temporal_downsample,
        )
    np.savez_compressed(
        item_dir / "predictions_public.npz",
        traces=traces.astype(np.float32),
        preds=preds.astype(np.float32),
        preds_raw=preds_raw.astype(np.float32),
        viz_indices=np.asarray(trial_ids, dtype=np.int64),
        frame_rates=np.asarray(frame_rates, dtype=np.float32),
    )
    summary = {
        "mat_path": str(mat_path),
        "n_trials": int(traces.shape[0]),
        "trace_length": int(traces.shape[1]),
        "frame_rate_median": frame_rate,
        "indicator_id": int(args.indicator_id),
        "temporal_downsample": int(args.temporal_downsample),
        "downsample_method": args.downsample_method,
        "trace_key": used_trace_key,
        "source_label": str(args.source_label),
        "max_frames_per_trace": int(args.max_frames_per_trace),
        "pred_stats": pred_stats(preds),
        "viz_indices": trial_ids,
    }
    (item_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[File Done] {mat_path} trials={traces.shape[0]} pred_mean={summary['pred_stats']['mean']:.3f}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, dm_hp = build_model(args.ckpt.resolve(), device)
    all_summary = {}
    for mat_path in args.input_files:
        mat_path = mat_path.resolve()
        all_summary[str(mat_path)] = run_file(args, model, dm_hp, mat_path)
    (args.output_dir / "summary_all.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[All Done] files={len(args.input_files)} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
