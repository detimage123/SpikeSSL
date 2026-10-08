#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cgi
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "spikessl_ui_default.ckpt"
RESULT_ROOT = ROOT / "result"
UPLOAD_ROOT = RESULT_ROOT / "spikessl_ui_uploads"
UPLOAD_RETENTION_HOURS = float(os.environ.get("SPIKESSL_UPLOAD_RETENTION_HOURS", "24"))
PUBLIC_SCRIPT = ROOT / "scripts" / "infer.py"
DEFAULT_GPU_ID = "0"
# The public release intentionally ships no bundled recordings. The UI accepts
# user uploads and deletes them after inference by default.
PRIVATE_DATA_ROOTS: list[Path] = []
TRACE_TYPES = [
    ("raw_trace_soma", "Raw Soma"),
    ("recon_trace_soma", "Recon Soma"),
    ("raw_trace_dend", "Raw Dend"),
    ("recon_trace_dend", "Recon Dend"),
]
SUPPORTED_UPLOAD_EXTENSIONS = {".mat", ".csv", ".npy", ".npz"}
STANDARD_TRACE_KEY = "raw_trace_soma"
METHOD_NAMES = {
    "dcvc_rt": "DCVC-RT",
    "vvc": "VVC",
    "nvrc": "NVRC",
    "pmd": "PMD",
    "necoda": "NeCoDA",
}
INDICATORS = [
    {"id": "auto", "name": "Auto Match Indicator"},
    {"id": 0, "name": "Unknown Indicator"},
    {"id": 1, "name": "OGB1"},
    {"id": 2, "name": "Cal520"},
    {"id": 3, "name": "GCaMP5k"},
    {"id": 4, "name": "GCaMP6f"},
    {"id": 5, "name": "GCaMP6s"},
    {"id": 6, "name": "GCaMP7f"},
    {"id": 7, "name": "GCaMP8f"},
    {"id": 8, "name": "GCaMP8m"},
    {"id": 9, "name": "GCaMP8s"},
    {"id": 10, "name": "R-CaMP"},
    {"id": 11, "name": "jRCaMP1a"},
    {"id": 12, "name": "jGECO1a"},
    {"id": 13, "name": "XCaMPgf"},
]
INDICATOR_NAMES = {str(item["id"]): item["name"] for item in INDICATORS}
PUBLIC_EXAMPLES = [
    {
        "label": "DS01 OGB1 mouse V1 cell 1",
        "path": str(ROOT / "data" / "DS01-OGB1-m-V1" / "CAttached_Theis16_set2_OGB_V1_cell_1_mini.mat"),
        "indicator_id": 1,
        "time_scale": 1,
    },
    {
        "label": "DS02 OGB1-2 mouse V1 L2/3 cell 1",
        "path": str(ROOT / "data" / "DS02-OGB1-2-m-V1" / "CAttached_Kwan2012_OGB_L23_pyramidal_cell1_mini.mat"),
        "indicator_id": 1,
        "time_scale": 1,
    },
    {
        "label": "DS03 Cal520 mouse S1 cell 2",
        "path": str(ROOT / "data" / "DS03-Cal520-m-S1" / "CAttached_S1_Cal520_cell2_mini.mat"),
        "indicator_id": 2,
        "time_scale": 1,
    },
    {
        "label": "DS05 Cal520 zebrafish pDp cell 2",
        "path": str(ROOT / "data" / "DS05-Cal520-zf-pDp" / "CAttached_Cal520_pDp_190118_Fish1_cell2_mini.mat"),
        "indicator_id": 2,
        "time_scale": 1,
    },
    {
        "label": "DS06 GCaMP6f zebrafish aDp cell 1",
        "path": str(ROOT / "data" / "DS06-GCaMP6f-zf-aDp" / "CAttached_GC_aDp_190501_Fish1_cell1_mini.mat"),
        "indicator_id": 4,
        "time_scale": 1,
    },
    {
        "label": "DS09 GCaMP6f mouse V1 cell 1",
        "path": str(ROOT / "data" / "DS09-GCaMP6f-m-V1" / "CAttached_Chen2013_GC6f_cell1_mini.mat"),
        "indicator_id": 4,
        "time_scale": 1,
    },
    {
        "label": "DS12 GCaMP6s mouse V1 neuropil corrected",
        "path": str(ROOT / "data" / "DS12-GCaMP6s-m-V1-neuropil-corrected" / "CAttached_Allen_tetO_101084_neuropil_subtracted_mini.mat"),
        "indicator_id": 5,
        "time_scale": 1,
    },
    {
        "label": "DS17 GCaMP5k mouse V1 cell 1",
        "path": str(ROOT / "data" / "DS17-GCaMP5k-m-V1" / "CAttached_Akerboom_GC5k_cell1_full_mini.mat"),
        "indicator_id": 3,
        "time_scale": 1,
    },
    {
        "label": "DS20 jRCaMP1a mouse V1 cell 1",
        "path": str(ROOT / "data" / "DS20-jRCaMP1a-m-V1" / "CAttached_Mohar16_jRCaMP1a_V1_1_mini.mat"),
        "indicator_id": 11,
        "time_scale": 1,
    },
    {
        "label": "DS29 GCaMP7f mouse V1 example",
        "path": str(ROOT / "data" / "DS29-GCaMP7f-m-V1" / "CAttached_jGCaMP7f_471991_1_mini.mat"),
        "indicator_id": 6,
        "time_scale": 1,
    },
    {
        "label": "DS30 GCaMP8f mouse V1 example",
        "path": str(ROOT / "data" / "DS30-GCaMP8f-m-V1" / "CAttached_jGCaMP8f_471993_1_mini.mat"),
        "indicator_id": 7,
        "time_scale": 1,
    },
    {
        "label": "DS31 GCaMP8m mouse V1 example",
        "path": str(ROOT / "data" / "DS31-GCaMP8m-m-V1" / "CAttached_jGCaMP8m_472179_2_mini.mat"),
        "indicator_id": 8,
        "time_scale": 1,
    },
    {
        "label": "DS32 GCaMP8s mouse V1 example",
        "path": str(ROOT / "data" / "DS32-GCaMP8s-m-V1" / "CAttached_jGCaMP8s_472181_1_mini.mat"),
        "indicator_id": 9,
        "time_scale": 1,
    },
]


def compression_label(path: Path) -> str:
    text = "/".join(path.parts[-3:])
    qp = re.search(r"qp(\d+)", text, flags=re.IGNORECASE)
    if qp:
        return f"QP{qp.group(1)}"
    cr = re.search(r"(?:^|[_/])cr(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if cr:
        value = float(cr.group(1))
        if value > 1000:
            value /= 100.0
        return f"CR {value:g}"
    return path.stem


def discover_private_examples() -> list[dict]:
    examples: list[dict] = []
    seen: set[str] = set()
    for private_root in PRIVATE_DATA_ROOTS:
        if not private_root.exists():
            continue
        for mat_path in sorted(private_root.glob("**/*.mat")):
            try:
                rel = mat_path.relative_to(private_root)
            except ValueError:
                rel = mat_path
            rel_key = str(rel)
            if rel_key in seen:
                continue
            seen.add(rel_key)
            method_key = rel.parts[0] if rel.parts else mat_path.parent.name
            method_name = METHOD_NAMES.get(method_key, method_key.upper())
            comp = compression_label(mat_path)
            for trace_key, trace_name in TRACE_TYPES:
                label = f"Private {method_name} {comp} - {trace_name}"
                examples.append(
                    {
                        "label": label,
                        "path": str(mat_path),
                        "indicator_id": "auto",
                        "time_scale": "auto",
                        "trace_key": trace_key,
                        "source_label": f"Private {method_name} {comp}",
                        "dataset_type": "private",
                    }
                )
    return examples


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def safe_upload_name(value: str) -> str:
    name = Path(value or "input.mat").name
    cleaned = safe_name(name)
    if not cleaned.lower().endswith(".mat"):
        cleaned += ".mat"
    return cleaned


def safe_upload_relative_path(value: str) -> Path:
    parts = []
    for part in Path(value or "input.mat").parts:
        cleaned = safe_name(part)
        if cleaned in {"", ".", ".."}:
            continue
        parts.append(cleaned)
    if not parts:
        parts = ["input.mat"]
    return Path(*parts)


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200, content_type: str = "text/html") -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", f"{content_type}; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def upload_session_dir_for(path: Path) -> Path | None:
    try:
        resolved = path.resolve()
        upload_root = UPLOAD_ROOT.resolve()
    except Exception:
        return None
    if resolved == upload_root or upload_root not in resolved.parents:
        return None
    rel = resolved.relative_to(upload_root)
    if len(rel.parts) < 2:
        return None
    return upload_root / rel.parts[0] / rel.parts[1]


def cleanup_uploaded_inputs(payload: dict, line_cb=None) -> None:
    keep = bool(payload.get("save_input_data"))
    if keep:
        if line_cb:
            line_cb("[Privacy] Uploaded input data kept because Save Uploaded Data is enabled.")
        return
    example_file = payload.get("example_file")
    if not example_file:
        return
    session_dir = upload_session_dir_for(Path(str(example_file)).expanduser())
    if not session_dir or not session_dir.exists():
        return
    shutil.rmtree(session_dir, ignore_errors=True)
    if line_cb:
        line_cb(f"[Privacy] Deleted uploaded input data: {session_dir}")


def cleanup_stale_uploads(max_age_hours: float = UPLOAD_RETENTION_HOURS) -> None:
    if max_age_hours <= 0 or not UPLOAD_ROOT.exists():
        return
    cutoff = time.time() - max_age_hours * 3600.0
    for run_dir in UPLOAD_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        for session_dir in run_dir.iterdir():
            if not session_dir.is_dir():
                continue
            try:
                mtime = session_dir.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)


class JobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.returncode: int | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.output_dir: Path | None = None
        self.command: list[str] = []
        self.lines: list[str] = []
        self.error = ""
        self.proc: subprocess.Popen | None = None
        self.phase = "Ready"
        self.analysis = ""

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = None
            if self.started_at is not None:
                elapsed = (self.finished_at or time.time()) - self.started_at
            return {
                "running": self.running,
                "returncode": self.returncode,
                "elapsed_sec": elapsed,
                "output_dir": str(self.output_dir) if self.output_dir else "",
                "command": " ".join(self.command),
                "lines": self.lines[-300:],
                "error": self.error,
                "phase": self.phase,
                "analysis": self.analysis,
            }

    def start(self, payload: dict, gpu_id: str = DEFAULT_GPU_ID) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.returncode = None
            self.started_at = time.time()
            self.finished_at = None
            self.output_dir = None
            self.command = []
            self.lines = []
            self.error = ""
            self.proc = None
            self.phase = "Analyzing signal timing"
            self.analysis = ""
        thread = threading.Thread(target=self._run, args=(payload, gpu_id), daemon=True)
        thread.start()
        return True

    def stop(self) -> bool:
        with self.lock:
            if not self.running or self.proc is None:
                return False
            proc = self.proc
        proc.terminate()
        return True

    def _append_line(self, line: str) -> None:
        with self.lock:
            self.lines.append(line.rstrip("\n"))
            if len(self.lines) > 2000:
                self.lines = self.lines[-1000:]

    def _set_analysis(self, text: str) -> None:
        with self.lock:
            self.analysis = text

    def _run(self, payload: dict, gpu_id: str) -> None:
        env = os.environ.copy()
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("PROJECT_ROOT", str(ROOT))
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_spikessl")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        returncode = -1
        try:
            command, output_dir, _ = build_command(payload, analysis_cb=self._set_analysis, line_cb=self._append_line)
            with self.lock:
                self.command = command
                self.output_dir = output_dir
                self.phase = "Running inference"
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            with self.lock:
                self.proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append_line(line)
            returncode = proc.wait()
            cleanup_uploaded_inputs(payload, line_cb=self._append_line)
        except Exception as exc:  # noqa: BLE001 - surfaced in UI
            returncode = -1
            with self.lock:
                self.error = str(exc)
            cleanup_uploaded_inputs(payload, line_cb=self._append_line)
        with self.lock:
            self.running = False
            self.returncode = returncode
            self.finished_at = time.time()
            self.proc = None
            self.phase = "Complete" if returncode == 0 else ("Paused" if returncode < 0 else "Error")


JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()


def normalize_run_id(value: str | None) -> str:
    text = str(value or "default")
    out = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
    return "".join(out)[:80] or "default"


def get_job(run_id: str) -> JobState:
    run_id = normalize_run_id(run_id)
    with JOBS_LOCK:
        if run_id not in JOBS:
            JOBS[run_id] = JobState()
        return JOBS[run_id]


def all_examples() -> list[dict]:
    normalized = []
    for item in PUBLIC_EXAMPLES:
        # Development examples are not part of the public repository. Retain
        # this optional hook only when a maintainer deliberately adds local
        # public examples to data/.
        if not Path(str(item["path"])).is_file():
            continue
        merged = dict(item)
        merged.setdefault("trace_key", "")
        merged.setdefault("source_label", "Public Example")
        merged.setdefault("dataset_type", "public")
        normalized.append(merged)
    normalized.extend(discover_private_examples())
    return normalized


def zip_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    result_root = RESULT_ROOT.resolve()
    if result_root not in output_dir.parents or not output_dir.is_dir():
        raise ValueError("No finished output directory is available.")
    zip_path = result_root / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir))
    return zip_path


def zip_visualization_images(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    result_root = RESULT_ROOT.resolve()
    if result_root not in output_dir.parents or not output_dir.is_dir():
        raise ValueError("No finished output directory is available.")
    zip_path = result_root / f"{output_dir.name}_images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*.png")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir))
    return zip_path


def ensure_2d(arr):
    import numpy as np

    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D trace array, got {out.shape}")
    return out


def orient_trace_matrix(arr):
    import numpy as np

    data = ensure_2d(arr)
    finite_rows = np.isfinite(data).any(axis=1)
    finite_cols = np.isfinite(data).any(axis=0)
    data = data[finite_rows][:, finite_cols]
    if data.size == 0:
        raise ValueError("Trace array contains no finite values.")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    rows, cols = data.shape
    transposed = False
    # Internal convention is [n_traces, n_frames]. Most calcium files have many
    # more frames than traces; if the opposite is true, treat columns as traces.
    if rows > cols:
        data = data.T
        transposed = True
    elif rows > 256 and cols < 256:
        data = data.T
        transposed = True
    if data.shape[1] < 16:
        raise ValueError(f"Trace length is too short after orientation: {data.shape}")
    return data.astype(np.float32, copy=False), transposed


def load_csv_traces(path: Path):
    import numpy as np

    data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 0 or data.size == 0:
        raise ValueError(f"No numeric trace data found in CSV: {path}")
    return orient_trace_matrix(data)


def load_npy_traces(path: Path):
    import numpy as np

    data = np.load(path, allow_pickle=False)
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"NPY array must be numeric: {path}")
    return orient_trace_matrix(data)


def load_npz_traces(path: Path):
    import numpy as np

    preferred = ["traces", STANDARD_TRACE_KEY, "raw_trace", "fluorescence", "dff", "data"]
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        ordered = [key for key in preferred if key in data] + [key for key in keys if key not in preferred]
        for key in ordered:
            arr = data[key]
            if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number):
                traces, transposed = orient_trace_matrix(arr)
                return traces, transposed, key
    raise ValueError(f"No numeric 1D/2D trace array found in NPZ: {path}")


def load_mat_traces_generic(path: Path):
    import numpy as np
    import scipy.io as sio

    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "CAttached" in data:
        traces, _, _ = load_public_mat_for_conversion(data["CAttached"])
        traces, transposed = orient_trace_matrix(traces)
        return traces, transposed, "CAttached.fluo_mean"

    for key in [name for name, _ in TRACE_TYPES] + ["traces", "trace", "fluorescence", "dff", "data"]:
        if key in data:
            traces, transposed = orient_trace_matrix(data[key])
            return traces, transposed, key

    candidates = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number) and arr.size >= 16:
            candidates.append((arr.ndim, arr.size, key, arr))
    if not candidates:
        raise ValueError(f"No numeric trace-like array found in MAT: {path}")
    _, _, key, arr = max(candidates, key=lambda item: (item[0], item[1]))
    traces, transposed = orient_trace_matrix(arr)
    return traces, transposed, key


def load_public_mat_for_conversion(cattached):
    import numpy as np

    trials = np.asarray(cattached, dtype=object).reshape(-1)
    traces = []
    frame_rates = []
    for trial_obj in trials:
        if not hasattr(trial_obj, "fluo_mean"):
            continue
        trace = np.squeeze(trial_obj.fluo_mean).astype(np.float32)
        if len(trace) < 16:
            continue
        traces.append(trace)
        if hasattr(trial_obj, "fluo_time"):
            times = np.squeeze(trial_obj.fluo_time).astype(np.float64)
            times = times[np.isfinite(times)]
            if len(times) > 2:
                rate = float(1.0 / np.mean(np.diff(times)))
                if np.isfinite(rate) and rate > 0:
                    frame_rates.append(rate)
    if not traces:
        raise ValueError("No usable CAttached.fluo_mean traces found.")
    min_len = min(len(trace) for trace in traces)
    return np.stack([trace[:min_len] for trace in traces], axis=0), [], frame_rates


def convert_input_to_standard_mat(source: Path, target_dir: Path) -> tuple[Path, dict]:
    import scipy.io as sio

    ext = source.suffix.lower()
    if ext == ".csv":
        traces, transposed = load_csv_traces(source)
        source_field = "csv_numeric_matrix"
    elif ext == ".npy":
        traces, transposed = load_npy_traces(source)
        source_field = "npy_array"
    elif ext == ".npz":
        traces, transposed, source_field = load_npz_traces(source)
    elif ext == ".mat":
        traces, transposed, source_field = load_mat_traces_generic(source)
    else:
        raise ValueError(f"Unsupported input format: {source.suffix}")

    rel_stem = safe_name(source.stem)
    target = target_dir / f"{rel_stem}.mat"
    counter = 1
    while target.exists():
        target = target_dir / f"{rel_stem}_{counter}.mat"
        counter += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "original_file": str(source),
        "original_format": ext.lstrip("."),
        "source_field": source_field,
        "orientation": "transposed_to_traces_by_frames" if transposed else "rows_as_traces",
        "n_traces": int(traces.shape[0]),
        "n_frames": int(traces.shape[1]),
        "standard_field": STANDARD_TRACE_KEY,
    }
    sio.savemat(
        target,
        {
            STANDARD_TRACE_KEY: traces,
            "spikessl_input_info_json": json.dumps(info, ensure_ascii=False),
        },
    )
    return target, info


def load_analysis_traces(path: Path, trace_key: str, max_traces: int = 24):
    import numpy as np
    import scipy.io as sio

    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "CAttached" in data:
        trials = np.asarray(data["CAttached"], dtype=object).reshape(-1)
        traces = []
        frame_rates = []
        for trial_obj in trials:
            if not hasattr(trial_obj, "fluo_mean"):
                continue
            trace = np.squeeze(trial_obj.fluo_mean).astype(np.float32)
            if len(trace) < 32:
                continue
            traces.append(trace)
            if hasattr(trial_obj, "fluo_time"):
                times = np.squeeze(trial_obj.fluo_time).astype(np.float64)
                times = times[np.isfinite(times)]
                if len(times) > 2:
                    rate = float(1.0 / np.mean(np.diff(times)))
                    if np.isfinite(rate) and rate > 0:
                        frame_rates.append(rate)
            if len(traces) >= max_traces:
                break
        if not traces:
            raise ValueError(f"No usable public traces found in {path}")
        min_len = min(len(t) for t in traces)
        return np.stack([t[:min_len] for t in traces], axis=0), frame_rates, "public"

    key = trace_key or "raw_trace_soma"
    if key not in data:
        for candidate, _ in TRACE_TYPES:
            if candidate in data:
                key = candidate
                break
    if key not in data:
        raise ValueError(f"No usable numeric trace field found in {path}")
    return ensure_2d(data[key])[:max_traces], [], "generic"


def merge_analysis_traces(paths: list[Path], trace_key: str, max_files: int = 8, max_traces_per_file: int = 6):
    import numpy as np

    traces_list = []
    frame_rates = []
    dataset_types = []
    used_files = []
    for path in paths[:max_files]:
        try:
            traces, rates, dataset_type = load_analysis_traces(path, trace_key, max_traces=max_traces_per_file)
        except Exception:
            continue
        if traces.size == 0:
            continue
        traces_list.append(ensure_2d(traces))
        frame_rates.extend(rates)
        dataset_types.append(dataset_type)
        used_files.append(path)
    if not traces_list:
        raise ValueError("No usable traces found for Auto analysis.")
    min_len = min(item.shape[1] for item in traces_list)
    merged = np.concatenate([item[:, :min_len] for item in traces_list], axis=0)
    dataset_type = "public" if dataset_types and all(item == "public" for item in dataset_types) else "generic"
    return merged, frame_rates, dataset_type, used_files


def zscore_rows(arr):
    import numpy as np

    data = ensure_2d(arr)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return np.where(std > 1e-6, (data - mean) / (std + 1e-6), data - mean)


def downsample_for_analysis(traces, factor: int):
    import numpy as np

    data = ensure_2d(traces)
    if factor <= 1:
        return data.astype(np.float32, copy=False)
    usable = (data.shape[1] // factor) * factor
    if usable < factor:
        return data.astype(np.float32, copy=False)
    return data[:, :usable].reshape(data.shape[0], -1, factor).mean(axis=2).astype(np.float32, copy=False)


def estimate_autocorr_half_lag(traces) -> float | None:
    import numpy as np

    values = []
    for trace in zscore_rows(traces[:24]):
        y = trace[: min(len(trace), 6000)]
        y = y - np.mean(y)
        if np.std(y) < 1e-6:
            continue
        max_lag = min(300, len(y) // 4)
        corr = np.correlate(y, y, mode="full")[len(y) - 1 : len(y) + max_lag]
        corr = corr / (corr[0] + 1e-8)
        below = np.where(corr < 0.5)[0]
        values.append(float(below[0]) if len(below) else float(max_lag))
    if not values:
        return None
    return float(np.median(values))


def estimate_peak_width(traces) -> float | None:
    import numpy as np

    widths = []
    kernel = np.ones(5, dtype=np.float32) / 5.0
    for trace in zscore_rows(traces[:24]):
        y = np.convolve(trace, kernel, mode="same")
        threshold = max(1.2, float(np.percentile(y, 90)))
        peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > threshold))[0] + 1
        if len(peaks) > 16:
            peaks = peaks[np.argsort(y[peaks])[-16:]]
        for peak in peaks:
            local = y[max(0, peak - 120) : min(len(y), peak + 120)]
            if len(local) < 8:
                continue
            base = float(np.percentile(local, 20))
            half = base + 0.5 * (float(y[peak]) - base)
            left = int(peak)
            while left > 0 and y[left] > half:
                left -= 1
            right = int(peak)
            while right < len(y) - 1 and y[right] > half:
                right += 1
            width = right - left
            if 3 <= width <= 1000:
                widths.append(float(width))
    if not widths:
        return None
    return float(np.median(widths))


def signal_features(traces) -> dict:
    import numpy as np

    z = zscore_rows(traces[:24])
    if z.size == 0:
        return {}
    dz = np.diff(z, axis=1)
    active_frac = float(np.mean(z > 1.0))
    diff_std = float(np.median(np.std(dz, axis=1))) if dz.size else 0.0
    high_q = float(np.median(np.percentile(z, 95, axis=1)))
    return {
        "ac_half": estimate_autocorr_half_lag(z),
        "peak_width": estimate_peak_width(z),
        "active_frac": active_frac,
        "diff_std": diff_std,
        "high_q": high_q,
    }


def feature_distance(a: dict, b: dict) -> float:
    scales = {
        "ac_half": 8.0,
        "peak_width": 14.0,
        "active_frac": 0.08,
        "diff_std": 0.25,
        "high_q": 0.8,
    }
    score = 0.0
    used = 0
    for key, scale in scales.items():
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            continue
        score += min(4.0, abs(float(av) - float(bv)) / scale)
        used += 1
    return score / max(1, used)


def feature_text(features: dict) -> str:
    parts = []
    if features.get("ac_half") is not None:
        parts.append(f"autocorr half-lag={features['ac_half']:.1f} frames")
    if features.get("peak_width") is not None:
        parts.append(f"transient width={features['peak_width']:.1f} frames")
    if features.get("active_frac") is not None:
        parts.append(f"active fraction={features['active_frac']:.3f}")
    return ", ".join(parts) if parts else "insufficient robust signal features"


REFERENCE_FEATURES: list[dict] | None = None


def reference_features() -> list[dict]:
    global REFERENCE_FEATURES
    if REFERENCE_FEATURES is not None:
        return REFERENCE_FEATURES
    refs = []
    for item in PUBLIC_EXAMPLES:
        try:
            traces, _, _ = load_analysis_traces(Path(item["path"]), "", max_traces=8)
            refs.append(
                {
                    "indicator_id": int(item["indicator_id"]),
                    "indicator_name": INDICATOR_NAMES.get(str(item["indicator_id"]), str(item["indicator_id"])),
                    "label": item["label"],
                    "features": signal_features(traces),
                }
            )
        except Exception:
            continue
    REFERENCE_FEATURES = refs
    return refs


def model_domain_target() -> dict:
    # This target is deliberately the model-domain scale, not a biological
    # indicator identity. This scale is used as a weak, model-domain prior for
    # time-scale selection when acquisition metadata are unavailable.
    return {"ac_half": 5.0, "peak_width": 11.0, "active_frac": 0.10, "diff_std": 0.22, "high_q": 1.7}


def choose_auto_time_scale(paths: list[Path], trace_key: str) -> tuple[int, str, dict]:
    import numpy as np

    traces, frame_rates, dataset_type, used_files = merge_analysis_traces(paths, trace_key)
    analyzed_text = f"analyzed {len(used_files)} file(s), {traces.shape[0]} trace(s)"
    candidates = [1, 2, 4, 5, 6, 8]
    median_frame_rate = float(np.median(frame_rates)) if frame_rates else None
    target = model_domain_target()

    if dataset_type == "public" and median_frame_rate and median_frame_rate > 75:
        factor = min(candidates, key=lambda x: abs((median_frame_rate / x) - 30.0))
        features = signal_features(downsample_for_analysis(traces, factor))
        analysis = (
            f"Time-scale analysis: public CAttached metadata gives median frame-rate={median_frame_rate:.1f} Hz. "
            f"Selected {factor}x to bring the effective rate near the model domain ({analyzed_text}); "
            f"measured {feature_text(features)} after normalization."
        )
        return factor, analysis, features

    if dataset_type == "public":
        features = signal_features(traces)
        analysis = (
            f"Time-scale analysis: public reference trace already matches the training-domain timing. "
            f"Selected 1x ({analyzed_text}); measured {feature_text(features)}."
        )
        return 1, analysis, features

    scored = []
    for candidate in candidates:
        features = signal_features(downsample_for_analysis(traces, candidate))
        score = feature_distance(features, target)
        # A weak model-domain prior favors factors that map broad calcium
        # transients to the temporal scale used during training.
        if candidate < 4:
            score += 0.65
        if candidate > 6:
            score += 0.35
        scored.append((score, candidate, features))
    score, factor, features = min(scored, key=lambda x: x[0])
    score_report = ", ".join(f"{candidate}x:{score:.2f}" for score, candidate, _ in scored)
    analysis = (
        "Time-scale analysis: trace has no reliable frame-rate metadata, so Auto compares candidate "
        f"downsampling factors against public training-domain trace dynamics. Selected {factor}x "
        f"(scores {score_report}; {analyzed_text}); measured {feature_text(features)} after normalization."
    )
    return int(factor), analysis, features


def choose_auto_indicator(paths: list[Path], trace_key: str, temporal_downsample: int) -> tuple[int, str]:
    traces, _, _, used_files = merge_analysis_traces(paths, trace_key)
    features = signal_features(downsample_for_analysis(traces, temporal_downsample))
    refs = reference_features()
    if not refs:
        return 0, "Indicator analysis: no public reference features were available; using Unknown Indicator (id=0)."

    scored = []
    for ref in refs:
        scored.append((feature_distance(features, ref["features"]), ref))
    scored.sort(key=lambda x: x[0])
    best_score, best_ref = scored[0]
    best_id = int(best_ref["indicator_id"])
    best_name = str(best_ref["indicator_name"])
    runner_up = "none"
    for runner_score, runner_ref in scored[1:]:
        if int(runner_ref["indicator_id"]) != best_id:
            runner_up = f"{runner_ref['indicator_name']} (distance={runner_score:.2f})"
            break
    analysis = (
        f"Indicator analysis: Auto compared normalized trace features with public indicator references. "
        f"Best match is {best_name} (id={best_id}, reference={best_ref['label']}, distance={best_score:.2f}; "
        f"runner-up={runner_up}; analyzed {len(used_files)} file(s), {traces.shape[0]} trace(s))."
    )
    return best_id, analysis


def resolve_input_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.mat") if p.is_file())
        if not files:
            raise ValueError(f"No .mat files found in input folder: {path}")
        return files
    if path.is_file():
        return [path]
    raise ValueError(f"Input path does not exist: {path}")


def build_command(payload: dict, analysis_cb=None, line_cb=None) -> tuple[list[str], Path, str]:
    selected_path = str(payload.get("example_file") or "").strip()
    if not selected_path:
        raise ValueError("Upload a fluorescence-trace file or folder before starting inference.")
    example_file = Path(selected_path).expanduser()
    input_files = resolve_input_files(example_file)
    ckpt = DEFAULT_CKPT
    gpu_id = DEFAULT_GPU_ID
    device = "cuda:0"
    indicator_value = str(payload.get("indicator_id") if payload.get("indicator_id") is not None else "0")
    trace_key = str(payload.get("trace_key") or "")
    source_label = str(payload.get("source_label") or "")
    temporal_value = str(payload.get("temporal_downsample") or "1")
    analysis_parts = []
    if temporal_value == "auto":
        temporal_downsample, time_analysis, normalized_features = choose_auto_time_scale(input_files, trace_key)
    else:
        temporal_downsample = max(1, int(float(temporal_value)))
        analysis_traces, _, _, used_files = merge_analysis_traces(input_files, trace_key)
        normalized_features = signal_features(downsample_for_analysis(analysis_traces, temporal_downsample))
        time_analysis = (
            f"Time-scale analysis: manual factor selected by user. Using {temporal_downsample}x; "
            f"analyzed {len(used_files)} file(s), {analysis_traces.shape[0]} trace(s); "
            f"measured {feature_text(normalized_features)} after normalization."
        )
    analysis_parts.append(time_analysis)

    if indicator_value == "auto":
        indicator_id, indicator_analysis = choose_auto_indicator(input_files, trace_key, temporal_downsample)
    else:
        indicator_id = int(float(indicator_value or 0))
        indicator_name = INDICATOR_NAMES.get(str(indicator_id), f"id={indicator_id}")
        indicator_analysis = f"Indicator analysis: manual selection {indicator_name} (id={indicator_id})."
    analysis_parts.append(indicator_analysis)
    analysis = " ".join(analysis_parts)
    if analysis_cb:
        analysis_cb(analysis)
    if line_cb:
        line_cb(f"[Analysis] {analysis}")
    downsample_method = "mean"
    viz_count = max(1, int(payload.get("viz_count") or 6))
    zoom_frames = max(16, int(payload.get("zoom_frames") or 500))
    max_frames_per_trace = max(0, int(payload.get("max_frames_per_trace") or 0))
    infer_batch_size = 1024
    input_transform = "zscore"
    run_name = safe_name(str(payload.get("run_name") or f"spikessl_ui_ds{temporal_downsample}_{now_tag()}"))
    output_dir = RESULT_ROOT / run_name

    command = [
        sys.executable,
        "-u",
        str(PUBLIC_SCRIPT),
        "--input-files",
        *[str(path) for path in input_files],
        "--output-dir",
        str(output_dir),
        "--ckpt",
        str(ckpt),
        "--device",
        device,
        "--indicator-id",
        str(indicator_id),
        "--temporal-downsample",
        str(temporal_downsample),
        "--downsample-method",
        downsample_method,
        "--input-transform",
        input_transform,
        "--viz-count",
        str(viz_count),
        "--zoom-frames",
        str(zoom_frames),
        "--infer-batch-size",
        str(infer_batch_size),
        "--max-trials-per-file",
        str(viz_count),
    ]
    if max_frames_per_trace > 0:
        command.extend(["--max-frames-per-trace", str(max_frames_per_trace)])
    if trace_key:
        command.extend(["--trace-key", trace_key])
    if source_label:
        command.extend(["--source-label", source_label])
    return command, output_dir, gpu_id


def gallery_for(output_dir: Path | None) -> list[dict]:
    if not output_dir or not output_dir.exists():
        return []
    items = []
    patterns = ["**/eval_style_public_spike.png", "**/eval_style_public_spike_part*.png"]
    paths = []
    for pattern in patterns:
        paths.extend(output_dir.glob(pattern))
    for path in sorted(paths):
        rel = path.relative_to(output_dir)
        items.append(
            {
                "label": str(rel.parent),
                "url": f"/file?path={path}",
            }
        )
    return items


def visualized_trace_count(output_dir: Path | None) -> int:
    if not output_dir or not output_dir.exists():
        return 0
    count = 0
    for summary_path in output_dir.glob("**/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        count += len(summary.get("viz_indices") or [])
    return count


def save_uploaded_files(handler: BaseHTTPRequestHandler, run_id: str) -> dict:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Upload must use multipart/form-data.")
    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
    )
    fields = form["files"] if "files" in form else []
    if not isinstance(fields, list):
        fields = [fields]
    fields = [field for field in fields if getattr(field, "filename", "")]
    if not fields:
        raise ValueError("No supported input files were uploaded.")

    upload_dir = UPLOAD_ROOT / normalize_run_id(run_id) / now_tag()
    original_dir = upload_dir / "original"
    converted_dir = upload_dir / "converted_mat"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    converted_paths = []
    conversion_infos = []
    for field in fields:
        rel_name = safe_upload_relative_path(field.filename)
        if rel_name.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            continue
        target = original_dir / rel_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            while True:
                chunk = field.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        saved_paths.append(target)
        converted, info = convert_input_to_standard_mat(target, converted_dir)
        converted_paths.append(converted)
        conversion_infos.append(info)
    if not converted_paths:
        raise ValueError("No supported .mat, .csv, .npy, or .npz files were uploaded.")

    if len(converted_paths) == 1:
        selected_path = converted_paths[0]
        info = conversion_infos[0]
        label = f"User Input - {Path(info['original_file']).name} ({info['n_traces']} traces x {info['n_frames']} frames)"
    else:
        selected_path = converted_dir
        label = f"User Input - {len(converted_paths)} files converted"
    return {
        "path": str(selected_path),
        "label": label,
        "source_label": label.replace(" - ", " "),
        "trace_key": STANDARD_TRACE_KEY,
        "count": len(converted_paths),
        "conversion": conversion_infos,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpikeSSL &mdash; Calcium Trace Deconvolution</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1a1f2e;
      --muted: #6b7280;
      --label: #374151;
      --line: #e2e5ea;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --accent-soft: #eff4ff;
      --green: #059669;
      --green-soft: #ecfdf5;
      --amber: #d97706;
      --amber-soft: #fffbeb;
      --red: #dc2626;
      --red-soft: #fef2f2;
      --radius: 10px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.06);
      --shadow-md: 0 4px 16px rgba(0,0,0,.07);
      --transition: .18s ease;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Inter", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Header ────────────────────────────────────────── */
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 20px 32px;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .header-spacer { flex: 1; }
    .brand-mark {
      display: inline-flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .logo-img {
      display: block;
      width: 44px;
      height: 44px;
      border-radius: 10px;
      object-fit: cover;
      flex: none;
      box-shadow: 0 2px 8px rgba(37, 99, 235, .18);
    }
    .brand-word {
      font-size: 22px;
      line-height: 1;
      font-weight: 780;
      color: var(--text);
      letter-spacing: 0;
      white-space: nowrap;
    }
    .brand-word .ssl {
      color: var(--accent);
    }
    .brand-text {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .brand-subtitle {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.15;
      white-space: nowrap;
    }
    .run-tabs-wrap {
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 10px 24px 0;
    }
    .run-tabs {
      max-width: 1560px;
      margin: 0 auto;
      display: flex;
      align-items: end;
      gap: 6px;
      overflow-x: auto;
    }
    .run-tab {
      height: 36px;
      padding: 0 8px 0 14px;
      border: 1px solid var(--line);
      border-bottom: none;
      border-radius: 8px 8px 0 0;
      background: #f8fafc;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .run-tab.active {
      background: var(--bg);
      color: var(--accent);
      border-color: #cbd5e1;
    }
    .run-tab-close {
      width: 18px;
      height: 18px;
      border: none;
      border-radius: 999px;
      background: transparent;
      color: #94a3b8;
      cursor: pointer;
      font-size: 16px;
      line-height: 18px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .run-tab-close:hover {
      background: #e2e8f0;
      color: #334155;
    }
    .run-tab-add {
      width: 34px;
      min-width: 34px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
      margin-bottom: 2px;
      font-size: 19px;
      line-height: 1;
    }
    .upload-input {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .upload-input .btn {
      width: 100%;
      justify-content: center;
      background: var(--accent);
      color: #fff;
      box-shadow: 0 1px 3px rgba(37,99,235,.25);
    }
    .upload-input .btn:hover:not(:disabled) {
      background: var(--accent-hover);
    }

    /* ── Layout ────────────────────────────────────────── */
    .container {
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 400px 1fr;
      gap: 24px;
      align-items: start;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow-sm);
    }
    .card-header {
      padding: 16px 20px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      font-weight: 650;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card-header .icon {
      width: 18px; height: 18px; fill: var(--accent); flex-shrink: 0;
    }
    .card-body { padding: 20px; }

    /* ── Form ──────────────────────────────────────────── */
    .field { margin-bottom: 18px; }
    .field:last-child { margin-bottom: 0; }
    .field-label {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--label);
      letter-spacing: 0;
      margin-bottom: 6px;
    }
    .field-hint {
      font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.4;
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 38px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafbfc;
      color: var(--label);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    .check-row input {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }
    select, input[type="number"] {
      width: 100%;
      height: 40px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font-size: 14px;
      font-family: inherit;
      transition: border var(--transition);
      appearance: auto;
    }
    select:focus, input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .compact-select {
      position: relative;
    }
    .compact-select select {
      color: transparent;
    }
    .compact-select select option {
      color: var(--text);
    }
    .compact-select-display {
      position: absolute;
      left: 12px;
      right: 34px;
      top: 50%;
      transform: translateY(-50%);
      pointer-events: none;
      color: var(--text);
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .field-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    /* ── Collapsible Advanced ──────────────────────────── */
    .advanced-toggle {
      margin-top: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .advanced-toggle summary {
      cursor: pointer;
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      background: #fafbfc;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 6px;
      user-select: none;
    }
    .advanced-toggle summary::-webkit-details-marker { display: none; }
    .advanced-toggle summary::before {
      content: "\25B6";
      font-size: 9px;
      transition: transform var(--transition);
    }
    .advanced-toggle[open] summary::before { transform: rotate(90deg); }
    .advanced-toggle .adv-body { padding: 14px; }

    /* ── Buttons ───────────────────────────────────────── */
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 22px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }
    .btn {
      height: 42px;
      border: none;
      border-radius: 8px;
      padding: 0 20px;
      font-size: 14px;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background var(--transition), box-shadow var(--transition);
    }
    .btn-primary {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 1px 3px rgba(37,99,235,.25);
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--accent-hover);
      box-shadow: 0 2px 8px rgba(37,99,235,.3);
    }
    .btn-primary:disabled {
      background: #93b4e8;
      cursor: not-allowed;
      box-shadow: none;
    }
    .btn-secondary {
      background: #f3f4f6;
      color: var(--text);
    }
    .btn-secondary:hover:not(:disabled) { background: #e5e7eb; }
    .btn-secondary:disabled { color: #9ca3af; cursor: not-allowed; }
    .btn-pause {
      background: #fff7ed;
      color: #9a3412;
      border: 1px solid #fed7aa;
    }
    .btn-pause:hover:not(:disabled) { background: #ffedd5; }
    .btn-pause:disabled { color: #cbd5e1; border-color: var(--line); background: #f8fafc; cursor: not-allowed; }
    .btn-icon {
      width: 38px;
      height: 38px;
      padding: 0;
      justify-content: center;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
    }
    .btn-icon:hover { background: #f3f4f6; }
    .tip-anchor,
    .btn[data-tip],
    .run-tab-add[data-tip] {
      position: relative;
    }
    .tip-anchor::after,
    .btn[data-tip]::after,
    .run-tab-add[data-tip]::after {
      content: attr(data-tip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 10px);
      transform: translate(-50%, 6px);
      z-index: 20;
      width: max-content;
      max-width: 280px;
      padding: 10px 12px;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      background: rgba(17, 24, 39, .96);
      color: #fff;
      font-size: 12px;
      line-height: 1.45;
      font-weight: 500;
      box-shadow: 0 12px 28px rgba(15, 23, 42, .18);
      opacity: 0;
      pointer-events: none;
      transition: opacity var(--transition), transform var(--transition);
      white-space: normal;
      text-align: left;
    }
    .tip-anchor::before,
    .btn[data-tip]::before,
    .run-tab-add[data-tip]::before {
      content: "";
      position: absolute;
      left: 50%;
      bottom: calc(100% + 3px);
      transform: translate(-50%, 6px);
      border: 7px solid transparent;
      border-top-color: rgba(17, 24, 39, .96);
      opacity: 0;
      pointer-events: none;
      transition: opacity var(--transition), transform var(--transition);
      z-index: 21;
    }
    .tip-anchor:hover::after,
    .tip-anchor:hover::before,
    .tip-anchor:focus-visible::after,
    .tip-anchor:focus-visible::before,
    .btn[data-tip]:hover::after,
    .btn[data-tip]:hover::before,
    .btn[data-tip]:focus-visible::after,
    .btn[data-tip]:focus-visible::before,
    .run-tab-add[data-tip]:hover::after,
    .run-tab-add[data-tip]:hover::before {
      opacity: 1;
      transform: translate(-50%, 0);
    }
    .help-tip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      color: var(--muted);
      background: #fff;
      font-size: 11px;
      font-weight: 800;
      cursor: help;
      flex: none;
    }

    /* ── Status Cards ──────────────────────────────────── */
    .status-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }
    .stat-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: #fafbfc;
      text-align: center;
    }
    .stat-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0;
      color: var(--muted);
    }
    .stat-value {
      margin-top: 4px;
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
    }
    .stat-value.status-running {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      line-height: 1;
    }
    .stat-card.state-running {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .stat-card.state-running .stat-value { color: var(--accent); }
    .running-mark {
      display: inline-flex;
      width: 28px;
      height: 24px;
      align-items: center;
      justify-content: center;
      flex: none;
    }
    .bot-head {
      position: relative;
      width: 24px;
      height: 18px;
      border-radius: 8px;
      background: linear-gradient(145deg, #60a5fa 0%, #2563eb 72%);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, .12), 0 7px 14px rgba(37, 99, 235, .22);
      animation: botFloat 1.8s ease-in-out infinite;
    }
    .bot-head::before {
      content: "";
      position: absolute;
      left: 50%;
      top: -6px;
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #38bdf8;
      box-shadow: 0 0 8px rgba(56, 189, 248, .85);
      transform: translateX(-50%);
      animation: botAntenna 1.8s ease-in-out infinite;
    }
    .bot-head::after {
      content: "";
      position: absolute;
      left: 7px;
      top: 8px;
      width: 4px;
      height: 4px;
      background: #fff;
      border-radius: 999px;
      box-shadow: 7px 0 0 #fff, 0 0 7px rgba(255,255,255,.9), 7px 0 7px rgba(255,255,255,.9);
      animation: botBlink 2.4s ease-in-out infinite;
    }
    @keyframes botFloat {
      0%, 100% { transform: translateY(0) rotate(0deg); }
      45% { transform: translateY(-2px) rotate(-2deg); }
      70% { transform: translateY(-1px) rotate(2deg); }
    }
    @keyframes botAntenna {
      0%, 100% { transform: translateX(-50%) scale(.9); }
      50% { transform: translateX(-50%) scale(1.15); }
    }
    @keyframes botBlink {
      0%, 80%, 100% { transform: scaleY(1); opacity: 1; }
      88% { transform: scaleY(.2); opacity: .85; }
    }
    .stat-card.state-done {
      border-color: var(--green);
      background: var(--green-soft);
    }
    .stat-card.state-done .stat-value { color: var(--green); }
    .stat-card.state-error {
      border-color: var(--red);
      background: var(--red-soft);
    }
    .stat-card.state-error .stat-value { color: var(--red); }
    .analysis-panel {
      display: none;
      margin: -8px 0 18px;
      padding: 12px;
      border: 1px solid #dbeafe;
      border-radius: 8px;
      background: #f8fbff;
      color: var(--text);
      font-size: 13px;
    }
    .analysis-panel.visible { display: block; }
    .analysis-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .analysis-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    .analysis-title {
      font-size: 12px;
      font-weight: 800;
      color: var(--label);
      margin-bottom: 8px;
    }
    .analysis-kv {
      display: grid;
      gap: 6px;
    }
    .analysis-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      line-height: 1.35;
    }
    .analysis-key {
      color: var(--muted);
      white-space: nowrap;
    }
    .analysis-value {
      color: var(--text);
      font-weight: 650;
      text-align: right;
    }
    .score-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .score-chip {
      padding: 3px 7px;
      border: 1px solid #dbeafe;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
    }
    .analysis-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    @media (max-width: 960px) {
      .analysis-grid { grid-template-columns: 1fr; }
    }

    /* ── Gallery ───────────────────────────────────────── */
    .gallery-empty {
      text-align: center;
      padding: 48px 20px;
      color: var(--muted);
      font-size: 14px;
    }
    .gallery-empty svg {
      width: 48px; height: 48px;
      fill: var(--line);
      margin-bottom: 12px;
    }
    .gallery {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .gallery figure {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      background: #fff;
      box-shadow: var(--shadow-sm);
      transition: box-shadow var(--transition);
    }
    .gallery figure:hover {
      box-shadow: var(--shadow-md);
    }
    .gallery figure img {
      display: block;
      width: 100%;
      height: auto;
      cursor: zoom-in;
    }
    .zoom-note {
      margin-bottom: 14px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafbfc;
      color: var(--muted);
      font-size: 13px;
    }

    /* ── Lightbox ──────────────────────────────────────── */
    .lightbox {
      position: fixed;
      inset: 0;
      z-index: 1000;
      background: rgba(0,0,0,.82);
      display: none;
      align-items: center;
      justify-content: center;
      cursor: zoom-out;
    }
    .lightbox.open { display: flex; }
    .lightbox img {
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 6px;
      box-shadow: 0 8px 40px rgba(0,0,0,.5);
    }

    /* ── Responsive ────────────────────────────────────── */
    @media (max-width: 960px) {
      .container { grid-template-columns: 1fr; }
      .status-grid { grid-template-columns: repeat(2, 1fr); }
      header { padding: 16px 20px; }
      .container { padding: 16px; }
    }
  </style>
</head>
<body>

  <header>
    <div class="brand-mark">
      <img class="logo-img" src="/logo.png" alt="SpikeSSL logo">
      <div class="brand-text">
        <div class="brand-word">Spike<span class="ssl">SSL</span></div>
        <div class="brand-subtitle">Calcium imaging trace-to-spike deconvolution</div>
      </div>
    </div>
    <div class="header-spacer"></div>
  </header>

  <div class="run-tabs-wrap">
    <div class="run-tabs" id="run_tabs"></div>
  </div>

  <div class="container">
    <!-- ─── Left: Settings ──────────────────────────── -->
    <div class="card">
      <div class="card-header">
        <svg class="icon" viewBox="0 0 24 24"><path d="M12 15.5A3.5 3.5 0 1 0 12 8.5a3.5 3.5 0 0 0 0 7Zm7.43-2.53c.04-.32.07-.64.07-.97s-.03-.66-.07-.97l2.11-1.65a.5.5 0 0 0 .12-.64l-2-3.46a.5.5 0 0 0-.61-.22l-2.49 1a7.06 7.06 0 0 0-1.67-.97l-.38-2.65A.49.49 0 0 0 14 2h-4a.49.49 0 0 0-.49.42l-.38 2.65c-.61.25-1.17.59-1.67.97l-2.49-1a.5.5 0 0 0-.61.22l-2 3.46a.49.49 0 0 0 .12.64l2.11 1.65c-.04.32-.07.65-.07.97s.03.66.07.97l-2.11 1.65a.5.5 0 0 0-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.5.38 1.06.72 1.67.97l.38 2.65c.05.24.26.42.49.42h4c.24 0 .44-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.67-.97l2.49 1c.22.08.49 0 .61-.22l2-3.46a.5.5 0 0 0-.12-.64l-2.11-1.65Z"/></svg>
        Settings
      </div>
      <div class="card-body">

        <div class="field">
          <label class="field-label">Input Trace
            <span class="help-tip tip-anchor" data-tip="Upload a fluorescence trace or a folder of traces. Supported formats are MAT, CSV, NPY, and NPZ.">?</span>
          </label>
          <select id="example_file"></select>
          <div class="upload-input">
            <button id="add_input_file" class="btn" type="button" data-tip="Select one or more local MAT, CSV, NPY, or NPZ files. Files are copied, converted to SpikeSSL trace format, and added as User Input.">
              + Add File
            </button>
            <button id="add_input_folder" class="btn" type="button" data-tip="Select a local folder. Supported MAT, CSV, NPY, and NPZ files inside the folder are converted and run together.">
              + Add Folder
            </button>
            <input id="input_file_picker" type="file" accept=".mat,.csv,.npy,.npz" multiple hidden>
            <input id="input_folder_picker" type="file" accept=".mat,.csv,.npy,.npz" multiple webkitdirectory directory hidden>
          </div>
        </div>

        <div class="field">
          <label class="field-label">Indicator
            <span class="help-tip tip-anchor" data-tip="Unknown Indicator uses id=0 directly. Auto Match Indicator compares trace dynamics with public indicator references and writes the selected indicator in the status analysis.">?</span>
          </label>
          <select id="indicator_id"></select>
        </div>

        <div class="field-row">
          <div class="field">
            <label class="field-label">Time-scale Factor
              <span class="help-tip tip-anchor" data-tip="Auto scores candidate time-scale factors by comparing normalized trace dynamics against the public training-domain scale. Manual values force a fixed factor.">?</span>
            </label>
            <select id="temporal_downsample">
              <option value="auto">Auto</option>
              <option value="1">1&times; (native)</option>
              <option value="2">2&times;</option>
              <option value="4">4&times;</option>
              <option value="5">5&times;</option>
              <option value="6" selected>6&times;</option>
              <option value="8">8&times;</option>
            </select>
          </div>
          <div class="field">
            <label class="field-label">Zoom Frames
              <span id="zoom_help" class="help-tip tip-anchor" data-tip="Number of processed frames shown in each zoom panel after time-scale normalization.">?</span>
            </label>
            <input id="zoom_frames" type="number" min="16" value="500">
          </div>
        </div>

        <input id="run_name" type="hidden" value="">

        <div class="field">
          <label class="check-row" data-tip="Server mode default: uploaded source files and converted temporary MAT files are deleted after inference. Results and figures remain downloadable.">
            <input id="save_input_data" type="checkbox">
            <span>Keep Uploaded Input Data On Server</span>
          </label>
        </div>

        <div class="field-row">
          <div class="field">
            <label class="field-label">Inference Range
              <span class="help-tip tip-anchor" data-tip="Preview modes crop each processed trace before sliding-window inference. Full Trace runs all windows and is slower.">?</span>
            </label>
            <div class="compact-select">
              <select id="max_frames_per_trace">
                <option value="0" selected>Full Trace - All Windows</option>
                <option value="2000">Fast Preview - First 2,000 Frames</option>
                <option value="5000">Medium Preview - First 5,000 Frames</option>
                <option value="10000">Long Preview - First 10,000 Frames</option>
              </select>
              <span id="max_frames_display" class="compact-select-display">Full Trace</span>
            </div>
          </div>
          <div class="field">
            <label class="field-label">Max Traces Shown
              <span class="help-tip tip-anchor" data-tip="Maximum traces to infer and visualize for this UI run. Smaller values are faster and reduce total sliding windows.">?</span>
            </label>
            <input id="viz_count" type="number" min="1" value="6">
          </div>
        </div>

        <div class="actions">
          <button id="run" class="btn btn-primary" type="button" data-tip="Analyze the selected trace timing, then run SpikeSSL inference on GPU0.">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            Run Inference
          </button>
          <button id="pause" class="btn btn-pause" type="button" disabled data-tip="Stop the current inference process. Finished files remain in the result folder.">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6V5zm8 0h4v14h-4V5z"/></svg>
            Pause
          </button>
          <button id="save" class="btn btn-secondary" type="button" disabled data-tip="Download all files from the current result folder as a zip archive.">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
            Save Results
          </button>
          <button id="save_images" class="btn btn-secondary" type="button" disabled data-tip="Download only visualization PNG files from the current result folder.">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>
            Save Images
          </button>
        </div>
      </div>
    </div>

    <!-- ─── Right: Results ──────────────────────────── -->
    <div class="card">
      <div class="card-header">
        <svg class="icon" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14H6v-2h6v2zm4-4H6v-2h10v2zm0-4H6V7h10v2z"/></svg>
        Results
      </div>
      <div class="card-body">

        <div class="status-grid">
          <div class="stat-card" id="state_card">
            <div class="stat-label">Status</div>
            <div class="stat-value" id="state">Ready</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Elapsed</div>
            <div class="stat-value" id="elapsed">&mdash;</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Visualized Traces</div>
            <div class="stat-value" id="fig_count">0</div>
          </div>
        </div>

        <div id="analysis_panel" class="analysis-panel"></div>

        <div class="zoom-note">Each result image has full-trace panels on the left and zoom panels on the right. Click the image to inspect the zoom columns at full size.</div>

        <div id="gallery">
          <div class="gallery-empty">
            <svg viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>
            <p>No results yet. Upload a trace and click <strong>Run Inference</strong>.</p>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- Lightbox overlay for image zoom -->
  <div class="lightbox" id="lightbox">
    <img id="lightbox_img" src="" alt="zoom">
  </div>

  <script>
    const examples = __EXAMPLES__;
    const indicators = __INDICATORS__;
    const params = new URLSearchParams(window.location.search);
    let runId = params.get("run_id");
    if (!runId) {
      runId = "run_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
      params.set("run_id", runId);
      window.history.replaceState(null, "", window.location.pathname + "?" + params.toString());
    }
    let runTabs = [{id: runId, label: "Run 1"}];
    const tabSettings = {};

    function apiUrl(path) {
      return path + "?run_id=" + encodeURIComponent(runId);
    }

    function currentSettings() {
      return {
        example_file: document.getElementById("example_file").value,
        indicator_id: document.getElementById("indicator_id").value,
        temporal_downsample: document.getElementById("temporal_downsample").value,
        max_frames_per_trace: document.getElementById("max_frames_per_trace").value,
        viz_count: document.getElementById("viz_count").value,
        zoom_frames: document.getElementById("zoom_frames").value,
      };
    }

    function saveTabSettings() {
      if (!runId || !document.getElementById("example_file").options.length) return;
      tabSettings[runId] = currentSettings();
    }

    function loadTabSettings(id) {
      const settings = tabSettings[id];
      if (!settings) {
        applyExampleDefaults();
        updateInferenceRangeDisplay();
        updateZoomHint();
        return;
      }
      for (const [key, value] of Object.entries(settings)) {
        const el = document.getElementById(key);
        if (el) el.value = value;
      }
      updateInferenceRangeDisplay();
      updateZoomHint();
    }

    function renderRunTabs() {
      const wrap = document.getElementById("run_tabs");
      wrap.innerHTML = "";
      runTabs.forEach(tab => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "run-tab" + (tab.id === runId ? " active" : "");
        const label = document.createElement("span");
        label.textContent = tab.label;
        btn.appendChild(label);
        if (runTabs.length > 1) {
          const close = document.createElement("span");
          close.className = "run-tab-close";
          close.textContent = "\u00d7";
          close.title = "Close this run tab";
          close.addEventListener("click", event => {
            event.stopPropagation();
            closeRunTab(tab.id);
          });
          btn.appendChild(close);
        }
        btn.addEventListener("click", () => activateRunTab(tab.id));
        wrap.appendChild(btn);
      });
      const add = document.createElement("button");
      add.type = "button";
      add.className = "run-tab-add";
      add.textContent = "+";
      add.dataset.tip = "Create a new run tab inside this UI. Existing results stay available by switching tabs.";
      add.addEventListener("click", createRunTab);
      wrap.appendChild(add);
    }

    function activateRunTab(id) {
      if (id === runId) return;
      saveTabSettings();
      runId = id;
      params.set("run_id", runId);
      window.history.replaceState(null, "", window.location.pathname + "?" + params.toString());
      loadTabSettings(runId);
      renderRunTabs();
      refresh();
    }

    function closeRunTab(id) {
      if (runTabs.length <= 1) return;
      const index = runTabs.findIndex(tab => tab.id === id);
      if (index < 0) return;
      delete tabSettings[id];
      runTabs.splice(index, 1);
      if (id === runId) {
        const next = runTabs[Math.max(0, index - 1)] || runTabs[0];
        runId = next.id;
        params.set("run_id", runId);
        window.history.replaceState(null, "", window.location.pathname + "?" + params.toString());
        loadTabSettings(runId);
        refresh();
      }
      renderRunTabs();
    }

    function createRunTab() {
      saveTabSettings();
      const nextId = "run_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
      runTabs.push({id: nextId, label: "Run " + (runTabs.length + 1)});
      runId = nextId;
      params.set("run_id", runId);
      window.history.replaceState(null, "", window.location.pathname + "?" + params.toString());
      tabSettings[runId] = currentSettings();
      renderRunTabs();
      refresh();
    }

    /* ── Populate dropdowns ──────────────────────────── */
    const indicatorSelect = document.getElementById("indicator_id");
    indicators.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = item.name;
      indicatorSelect.appendChild(opt);
    });

      const exampleSelect = document.getElementById("example_file");
      examples.forEach((item, idx) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.textContent = item.label;
      opt.dataset.indicatorId = item.indicator_id;
      opt.dataset.timeScale = item.time_scale;
      opt.dataset.traceKey = item.trace_key || "";
      opt.dataset.sourceLabel = item.source_label || item.label;
      opt.dataset.datasetType = item.dataset_type || "public";
      if (idx === 0) opt.selected = true;
      exampleSelect.appendChild(opt);
    });

    function shortPathLabel(path) {
      const clean = String(path || "").replace(/\/+$/, "");
      const parts = clean.split(/[\\/]/).filter(Boolean);
      return parts.length ? parts[parts.length - 1] : clean;
    }

    function addUserInputOption(item) {
      const value = item.path;
      const existing = Array.from(exampleSelect.options).find(opt => opt.value === value && opt.dataset.datasetType === "user");
      if (existing) {
        existing.selected = true;
        applyExampleDefaults();
        return;
      }
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = item.label || ("User Input - " + shortPathLabel(value));
      opt.dataset.indicatorId = "auto";
      opt.dataset.timeScale = "auto";
      opt.dataset.traceKey = item.trace_key || "raw_trace_soma";
      opt.dataset.sourceLabel = item.source_label || ("User Input " + shortPathLabel(value));
      opt.dataset.datasetType = "user";
      exampleSelect.appendChild(opt);
      opt.selected = true;
      applyExampleDefaults();
    }

    function addInputFileOption() {
      document.getElementById("input_file_picker").click();
    }

    function addInputFolderOption() {
      document.getElementById("input_folder_picker").click();
    }

    async function uploadInputFiles(pickerId) {
      const picker = document.getElementById(pickerId);
      const supportedExts = [".mat", ".csv", ".npy", ".npz"];
      const files = Array.from(picker.files || []).filter(file => supportedExts.some(ext => file.name.toLowerCase().endsWith(ext)));
      if (files.length === 0) return;
      const form = new FormData();
      files.forEach(file => form.append("files", file, file.webkitRelativePath || file.name));
      const res = await fetch(apiUrl("/api/upload"), { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        alert(data.error || "Unable to add input data.");
        picker.value = "";
        return;
      }
      addUserInputOption(data.item);
      picker.value = "";
    }

    function applyExampleDefaults() {
      const opt = exampleSelect.selectedOptions[0];
      if (!opt) return;
      document.getElementById("indicator_id").value = opt.dataset.indicatorId || "0";
      document.getElementById("temporal_downsample").value = opt.dataset.timeScale || "1";
      updateZoomHint();
    }

    function updateZoomHint() {
      const value = document.getElementById("temporal_downsample").value || "1";
      const zoomHelp = document.getElementById("zoom_help");
      if (value === "auto") {
        zoomHelp.dataset.tip = "Zoom frames are counted after Auto selects and applies the time-scale factor.";
        return;
      }
      const factor = Math.max(1, Number(value || 1));
      const zoom = Math.max(0, Number(document.getElementById("zoom_frames").value || 0));
      const rawFrames = Math.round(zoom * factor);
      zoomHelp.dataset.tip = `Zoom uses ${zoom} processed frames after time-scale normalization, about ${rawFrames} raw frames at ${factor}x.`;
    }

    function updateInferenceRangeDisplay() {
      const select = document.getElementById("max_frames_per_trace");
      const text = select.selectedOptions[0]?.textContent || "";
      document.getElementById("max_frames_display").textContent = text.split(" - ")[0] || text;
    }

    /* ── Payload ─────────────────────────────────────── */
    function getPayload() {
      const opt = exampleSelect.selectedOptions[0];
      return {
        example_file: document.getElementById("example_file").value,
        indicator_id: document.getElementById("indicator_id").value || "0",
        temporal_downsample: document.getElementById("temporal_downsample").value || "1",
        max_frames_per_trace: Number(document.getElementById("max_frames_per_trace").value || 0),
        viz_count: Number(document.getElementById("viz_count").value || 6),
        zoom_frames: Number(document.getElementById("zoom_frames").value || 500),
        trace_key: opt?.dataset.traceKey || "",
        source_label: opt?.dataset.sourceLabel || "",
        dataset_type: opt?.dataset.datasetType || "public",
        save_input_data: document.getElementById("save_input_data").checked,
        run_name: document.getElementById("run_name").value,
        run_id: runId,
      };
    }

    /* ── Run ──────────────────────────────────────────── */
    async function runJob() {
      document.getElementById("run").disabled = true;
      const res = await fetch(apiUrl("/api/run"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(getPayload()),
      });
      const data = await res.json();
      if (!res.ok) alert(data.error || "Inference failed to start.");
      await refresh();
    }

    function saveResults() {
      window.location.href = apiUrl("/download_results");
    }

    async function pauseJob() {
      const res = await fetch(apiUrl("/api/stop"), {method: "POST"});
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        alert(data.error || "Unable to pause the current run.");
      }
      await refresh();
    }

    function saveImages() {
      window.location.href = apiUrl("/download_images");
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function matchOne(text, regex, fallback = "") {
      const match = text.match(regex);
      return match ? match[1] : fallback;
    }

    function analysisRow(key, value) {
      if (!value) return "";
      return `<div class="analysis-row"><span class="analysis-key">${escapeHtml(key)}</span><span class="analysis-value">${escapeHtml(value)}</span></div>`;
    }

    function renderAnalysis(text) {
      if (!text) return "";
      const selected = matchOne(text, /Selected\s+(\d+x)/);
      const scores = matchOne(text, /scores\s+([^)；;]+?)(?:;\s*analyzed|\))/);
      const analyzedTime = matchOne(text, /scores\s+[^)]*?;\s*(analyzed\s+\d+\s+file\(s\),\s+\d+\s+trace\(s\))/);
      const ac = matchOne(text, /autocorr half-lag=([\d.]+\s+frames)/);
      const width = matchOne(text, /transient width=([\d.]+\s+frames)/);
      const active = matchOne(text, /active fraction=([\d.]+)/);
      const bestMatch = text.match(/Best match is\s+(.+?)\s+\(id=(\d+),\s+reference=(.+?),\s+distance=([\d.]+);\s+runner-up=(.+?)(?:;\s*(analyzed\s+\d+\s+file\(s\),\s+\d+\s+trace\(s\)))?\)\./);
      const scoreChips = scores
        ? `<div class="score-chips">${scores.split(",").map(item => `<span class="score-chip">${escapeHtml(item.trim())}</span>`).join("")}</div>`
        : "";
      const indicatorRows = bestMatch
        ? [
            analysisRow("Best Match", bestMatch[1]),
            analysisRow("Indicator ID", bestMatch[2]),
            analysisRow("Reference", bestMatch[3]),
            analysisRow("Distance", bestMatch[4]),
            analysisRow("Runner-up", bestMatch[5]),
            analysisRow("Data Used", bestMatch[6] || "")
          ].join("")
        : `<div class="analysis-note">${escapeHtml(text)}</div>`;
      const hasStructured = selected || bestMatch;
      if (!hasStructured) return `<div class="analysis-note">${escapeHtml(text)}</div>`;
      return `
        <div class="analysis-grid">
          <div class="analysis-card">
            <div class="analysis-title">Time-scale Analysis</div>
            <div class="analysis-kv">
              ${analysisRow("Selected Factor", selected)}
              ${analysisRow("Data Used", analyzedTime)}
              ${analysisRow("Autocorr Half-lag", ac)}
              ${analysisRow("Transient Width", width)}
              ${analysisRow("Active Fraction", active)}
            </div>
            ${scoreChips}
          </div>
          <div class="analysis-card">
            <div class="analysis-title">Indicator Analysis</div>
            <div class="analysis-kv">${indicatorRows}</div>
          </div>
        </div>`;
    }

    /* ── Helpers ──────────────────────────────────────── */
    function sec(value) {
      if (value === null || value === undefined) return "\u2014";
      if (value < 60) return Math.round(value) + "s";
      return Math.floor(value / 60) + "m " + Math.round(value % 60) + "s";
    }

    /* ── Lightbox ─────────────────────────────────────── */
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox_img");
    lightbox.addEventListener("click", () => lightbox.classList.remove("open"));
    function openLightbox(src) {
      lightboxImg.src = src;
      lightbox.classList.add("open");
    }

    /* ── Refresh / Poll ───────────────────────────────── */
    let pollTimer = null;
    async function refresh() {
      try {
        const res = await fetch(apiUrl("/api/status"));
        const data = await res.json();

        const stateEl = document.getElementById("state");
        const stateCard = document.getElementById("state_card");
        const runBtn = document.getElementById("run");
        const pauseBtn = document.getElementById("pause");
        const saveBtn = document.getElementById("save");
        const saveImagesBtn = document.getElementById("save_images");
        const analysisPanel = document.getElementById("analysis_panel");

        stateCard.className = "stat-card";
        stateEl.classList.remove("status-running");
        if (data.running) {
          const phase = data.phase || "Running inference";
          const label = phase.includes("Analyzing") ? "Analyzing" : "Running";
          stateEl.classList.add("status-running");
          stateEl.innerHTML = '<span class="running-mark" aria-hidden="true"><span class="bot-head"></span></span><span>' + label + '</span>';
          stateCard.classList.add("state-running");
          runBtn.disabled = true;
          pauseBtn.disabled = false;
          saveBtn.disabled = true;
          saveImagesBtn.disabled = true;
        } else if (data.returncode !== null && data.returncode !== undefined) {
          if (data.returncode === 0) {
            stateEl.textContent = "Complete";
            stateCard.classList.add("state-done");
          } else if (data.returncode < 0) {
            stateEl.textContent = "Paused";
            stateCard.classList.add("state-error");
          } else {
            stateEl.textContent = "Error";
            stateCard.classList.add("state-error");
          }
          runBtn.disabled = false;
          pauseBtn.disabled = true;
          saveBtn.disabled = !data.output_dir;
          saveImagesBtn.disabled = !data.output_dir;
        } else {
          stateEl.textContent = "Ready";
          runBtn.disabled = false;
          pauseBtn.disabled = true;
          saveBtn.disabled = true;
          saveImagesBtn.disabled = true;
        }
        if (data.analysis) {
          analysisPanel.innerHTML = renderAnalysis(data.analysis);
          analysisPanel.classList.add("visible");
        } else {
          analysisPanel.innerHTML = "";
          analysisPanel.classList.remove("visible");
        }

        document.getElementById("elapsed").textContent = sec(data.elapsed_sec);

        const galleryRes = await fetch(apiUrl("/api/gallery"));
        const gallery = await galleryRes.json();
        document.getElementById("fig_count").textContent = gallery.trace_count || 0;

        const node = document.getElementById("gallery");
        if (gallery.items.length === 0) {
          node.innerHTML = '<div class="gallery-empty">' +
            '<svg viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>' +
            '<p>No results yet. Upload a trace and click <strong>Run Inference</strong>.</p></div>';
        } else {
          node.innerHTML = "";
          node.className = "gallery";
          gallery.items.forEach(item => {
            const fig = document.createElement("figure");
            const img = document.createElement("img");
            img.src = item.url;
            img.alt = item.label;
            img.addEventListener("click", () => openLightbox(item.url));
            fig.appendChild(img);
            node.appendChild(fig);
          });
        }
      } catch (_) {}
    }

    /* ── Event wiring ─────────────────────────────────── */
    document.getElementById("run").addEventListener("click", runJob);
    document.getElementById("pause").addEventListener("click", pauseJob);
    document.getElementById("save").addEventListener("click", saveResults);
    document.getElementById("save_images").addEventListener("click", saveImages);
    document.getElementById("add_input_file").addEventListener("click", addInputFileOption);
    document.getElementById("add_input_folder").addEventListener("click", addInputFolderOption);
    document.getElementById("input_file_picker").addEventListener("change", () => uploadInputFiles("input_file_picker"));
    document.getElementById("input_folder_picker").addEventListener("change", () => uploadInputFiles("input_folder_picker"));
    document.getElementById("example_file").addEventListener("change", applyExampleDefaults);
    document.getElementById("temporal_downsample").addEventListener("change", updateZoomHint);
    document.getElementById("zoom_frames").addEventListener("input", updateZoomHint);
    document.getElementById("max_frames_per_trace").addEventListener("change", updateInferenceRangeDisplay);
    applyExampleDefaults();
    updateZoomHint();
    updateInferenceRangeDisplay();
    renderRunTabs();

    /* Poll every 1s but DO NOT trigger inference */
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


def render_index() -> str:
    return (
        HTML.replace("__EXAMPLES__", json.dumps(all_examples(), ensure_ascii=False))
        .replace("__INDICATORS__", json.dumps(INDICATORS, ensure_ascii=False))
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        run_id = normalize_run_id(query.get("run_id", ["default"])[0])
        if parsed.path == "/":
            text_response(self, render_index())
            return
        if parsed.path == "/logo.png":
            logo_path = ROOT / "webui" / "logo.png"
            if not logo_path.exists():
                self.send_error(404)
                return
            data = logo_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/status":
            json_response(self, get_job(run_id).snapshot())
            return
        if parsed.path == "/api/gallery":
            snap = get_job(run_id).snapshot()
            output_dir = Path(snap["output_dir"]) if snap.get("output_dir") else None
            json_response(self, {"items": gallery_for(output_dir), "trace_count": visualized_trace_count(output_dir)})
            return
        if parsed.path == "/download_results":
            snap = get_job(run_id).snapshot()
            output_dir = Path(snap["output_dir"]) if snap.get("output_dir") else None
            if not output_dir or snap.get("running"):
                self.send_error(404)
                return
            try:
                zip_path = zip_output_dir(output_dir)
            except Exception as exc:  # noqa: BLE001
                self.send_error(400, str(exc))
                return
            data = zip_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{zip_path.name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/download_images":
            snap = get_job(run_id).snapshot()
            output_dir = Path(snap["output_dir"]) if snap.get("output_dir") else None
            if not output_dir or snap.get("running"):
                self.send_error(404)
                return
            try:
                zip_path = zip_visualization_images(output_dir)
            except Exception as exc:  # noqa: BLE001
                self.send_error(400, str(exc))
                return
            data = zip_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{zip_path.name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/file":
            path_value = query.get("path", [""])[0]
            path = Path(path_value)
            if not path.exists() or not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        run_id = normalize_run_id(query.get("run_id", ["default"])[0])
        if parsed.path == "/api/stop":
            ok = get_job(run_id).stop()
            if not ok:
                json_response(self, {"error": "No running job to pause."}, status=409)
                return
            json_response(self, {"ok": True})
            return
        if parsed.path == "/api/upload":
            try:
                item = save_uploaded_files(self, run_id)
            except Exception as exc:  # noqa: BLE001
                json_response(self, {"error": str(exc)}, status=400)
                return
            json_response(self, {"ok": True, "item": item})
            return
        if parsed.path != "/api/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        run_id = normalize_run_id(payload.get("run_id") or query.get("run_id", ["default"])[0])
        ok = get_job(run_id).start(payload, gpu_id=DEFAULT_GPU_ID)
        if not ok:
            json_response(self, {"error": "A job is already running."}, status=409)
            return
        json_response(self, {"ok": True})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[webui] {self.address_string()} {fmt % args}", flush=True)


def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Web UI for SpikeSSL inference.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_server_args()
    cleanup_stale_uploads()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SpikeSSL Web UI: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
