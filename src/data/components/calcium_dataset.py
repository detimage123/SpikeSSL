import os
import glob
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from pynwb import NWBHDF5IO

from src.utils import pylogger
from src.utils.signal_processing import denoise_signal, smooth_spikes, resample_trace
from src.models.components.spikessm_mm import (
    parse_indicator_id, parse_species_id, parse_tissue_id, parse_cell_type_id,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class CalciumDataset(Dataset):

    def __init__(
        self,
        data_paths: List[str], 
        sampling_rate: float,
        smoothing: float,
        segment_len: int,
        context_left: int = 30,
        context_right: int = 30,
        overlap_size: int = 2,
        resample: bool = False,
        omission_dict: Dict[str, List[int]] = None,
        causal_kernel: bool = False,
        denoise_method: str = "none",
        denoise_params: dict = None,
        feature_mode: str = "dual",
        ephys_fs: float = 1e4,
        global_norm: str = "none",   
        bad_data_dict: Dict[str, Dict[str, set]] = None,  # {ds_name: {mat_file: {0-based trial indices}}}
        target_clip: float = None,   # if set, clip spike targets to [0, target_clip] (e.g. 1.0 for OGB1)
    ):
        super().__init__()

        # self.data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.data: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
        self.segment_len = segment_len
        self.segment_total_len = segment_len + context_left + context_right
        self.stride = max(1, self.segment_total_len // overlap_size)
        self.context_left = context_left
        self.context_right = context_right

        self.target_sampling_rate = sampling_rate
        self.smoothing = smoothing
        self.resample = resample
        self.causal_kernel = causal_kernel
        self.denoise_method = denoise_method
        self.denoise_params = denoise_params if denoise_params is not None else {}
        self.ephys_fs = ephys_fs        
        self.target_clip = target_clip   # optional upper bound for spike targets
        
        self.omission_dict = omission_dict if omission_dict is not None else {}
        self.bad_data_dict = bad_data_dict if bad_data_dict is not None else {}
        
        valid_modes = ["raw", "diff", "dual"]
        if feature_mode not in valid_modes:
            raise ValueError(f"The feature_mode is: {feature_mode}. Must be one of {valid_modes}")
        self.feature_mode = feature_mode

        valid_gnorm = ["none", "zscore", "minmax"]
        if global_norm not in valid_gnorm:
            raise ValueError(f"global_norm must be one of {valid_gnorm}, got '{global_norm}'")
        self.global_norm = global_norm

        
        self._gnorm_mean: float = 0.0
        self._gnorm_std:  float = 1.0
        self._gnorm_min:  float = -4.0
        self._gnorm_max:  float = 4.0

        
        files_to_load = []
        
        for path in data_paths:
            if not os.path.exists(path):
                log.warning(f"Path not found: {path}")
                continue
            
            if os.path.isfile(path):
                if path.endswith('.mat'):
                    files_to_load.append(path)
                elif path.endswith('.nwb'):
                    files_to_load.append(path)
                else:
                    log.warning(f"Skipping unsupported file: {path}")
            
            elif os.path.isdir(path):
                self._load_folder(path)

        if files_to_load:
            self._load_files(files_to_load)

        if len(self.data) == 0:
            log.warning("Dataset is empty! Check your data paths or omission lists.")

        
        if self.global_norm != "none" and len(self.data) > 0:
            sample_n = min(len(self.data), 50000)
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(self.data), sample_n, replace=False)
            flat = np.concatenate([self.data[i][0].reshape(-1) for i in idx])
            if self.global_norm == "zscore":
                self._gnorm_mean = float(flat.mean())
                self._gnorm_std  = float(flat.std()) if flat.std() > 1e-8 else 1.0
                log.info(f"global_norm=zscore  mean={self._gnorm_mean:.4f}  std={self._gnorm_std:.4f}")
            elif self.global_norm == "minmax":
                self._gnorm_min = float(np.percentile(flat, 1))   
                self._gnorm_max = float(np.percentile(flat, 99))  # 99th percentile
                log.info(f"global_norm=minmax  p1={self._gnorm_min:.4f}  p99={self._gnorm_max:.4f}")


    def __getitem__(self, idx):
        trace_window, rate_window, meta = self.data[idx]

        
        if self.global_norm == "zscore":
            trace_window = (trace_window - self._gnorm_mean) / self._gnorm_std
        elif self.global_norm == "minmax":
            rng = self._gnorm_max - self._gnorm_min
            trace_window = np.clip(
                (trace_window - self._gnorm_min) / (rng + 1e-8), 0.0, 1.0
            )

        y = torch.tensor(rate_window, dtype=torch.float32).unsqueeze(-1)
        
        if self.feature_mode == "raw":
            x_output = torch.tensor(trace_window, dtype=torch.float32).unsqueeze(-1)
        elif self.feature_mode == "diff":
            diff_feat = np.diff(trace_window, prepend=trace_window[0])
            x_output = torch.tensor(diff_feat, dtype=torch.float32).unsqueeze(-1)
        elif self.feature_mode == "dual":
            diff_feat = np.diff(trace_window, prepend=trace_window[0])
            x_output = torch.stack([
                torch.tensor(trace_window, dtype=torch.float32),
                torch.tensor(diff_feat, dtype=torch.float32)
            ], dim=1)

        weight = torch.ones_like(y) 
        if self.context_left > 0:
            weight[:self.context_left, :] = 0.0
        if self.context_right > 0:
            weight[-self.context_right:, :] = 0.0
        
        return x_output, y, weight, meta


    def __len__(self):
        return len(self.data)
    

    '''
    def _load_folder(self, folder: str):
        """
        Load all MAT and NWB files within a folder.
        """

        file_list = []
        file_list.extend(glob.glob(os.path.join(folder, '**', '*_mini.mat'), recursive=True))
        file_list.extend(glob.glob(os.path.join(folder, '**', '*.nwb'), recursive=True))
        
        if not file_list:
            return

        folder_to_files = {}
        for f in file_list:
            parent_dir = os.path.dirname(f)
            if parent_dir not in folder_to_files:
                folder_to_files[parent_dir] = []
            folder_to_files[parent_dir].append(f)

        valid_files = []
        for parent_dir, flist in folder_to_files.items():
            flist = sorted(flist)
            folder_name = os.path.basename(parent_dir)
            files_omission = self.omission_dict.get(folder_name, [])
            
            valid_files.extend([f for i, f in enumerate(flist) if i not in files_omission])
            
        self._load_files(valid_files)
    '''

    
    def _load_folder(self, folder: str):
        ""





        file_list = []
        file_list.extend(glob.glob(os.path.join(folder, '*_mini.mat')))
        file_list.extend(glob.glob(os.path.join(folder, '*.nwb')))
        
        # Simulated data uses *.mat without _mini suffix
        if not file_list and "Simulated" in os.path.basename(folder):
            file_list.extend(glob.glob(os.path.join(folder, '*.mat')))

        file_list = sorted(file_list) 
        
        folder_name = os.path.basename(folder)
        files_omission = self.omission_dict.get(folder_name, [])
        
        valid_files = [f for i, f in enumerate(file_list) if i not in files_omission]
        self._load_files(valid_files, folder_name=folder_name)
    


    def _load_files(self, file_paths: List[str], folder_name: str = ""):
        ""

        for file_path in file_paths:
            try:
                if file_path.endswith('.mat'):
                    self._process_mat_data(file_path, folder_name=folder_name)
                elif file_path.endswith('.nwb'):
                    self._process_nwb_data(file_path, folder_name=folder_name)
            except Exception as e:
                log.warning(f"Failed to load {file_path}: {e}")


    @staticmethod
    def _unwrap_mat_field(field: np.ndarray) -> np.ndarray:
        """Unwrap MATLAB cell-array fields saved as dtype=object.

        Some datasets (e.g. DS07, DS19) store fluo_mean / fluo_time as
        nested cell arrays: shape=(1,1), dtype=object, with the actual
        float array inside. One .flat[0] extracts it.
        """
        if field.dtype == object:
            field = field.flat[0]
        return np.squeeze(field)

    def _process_mat_data(self, file_path: str, folder_name: str = ""):
        ""




        file_name = os.path.basename(file_path)
        dataset_neuron_all = sio.loadmat(file_path)['CAttached'][0]

        for i, trial in enumerate(dataset_neuron_all):
            # --- bad_data trial-level filtering (0-based) ---
            bad_trials = self.bad_data_dict.get(folder_name, {}).get(file_name, set())
            if i in bad_trials:
                continue

            try:
                keys = trial[0][0].dtype.descr
                keys_unfolded = list(sum(keys, ()))
                
                try:
                    traces_idx = int(keys_unfolded.index("fluo_mean") / 2)
                    time_idx = int(keys_unfolded.index("fluo_time") / 2)
                    event_idx = int(keys_unfolded.index("events_AP") / 2)
                except ValueError:
                    continue

                events = self._unwrap_mat_field(trial[0][0][event_idx])
                events = events[~np.isnan(events)] if events.ndim > 0 else np.array([])
                fluo_times = self._unwrap_mat_field(trial[0][0][time_idx])
                traces_mean = self._unwrap_mat_field(trial[0][0][traces_idx])

                trial_id = f"{file_name}_trial{i:02d}"

                # Simulated data: events are already in seconds (ephys_fs=1.0)
                current_ephys_fs = 1.0 if "Simulated" in folder_name else self.ephys_fs

                
                self._process_signal_to_windows(traces_mean, fluo_times, events, current_ephys_fs, trial_id=trial_id, folder_name=folder_name)

            except Exception as e:
                log.warning(f"Skipping trial {i} in {os.path.basename(file_path)}: {e}")
                continue


    def _process_nwb_data(self, file_path: str, folder_name: str = ""):
        ""





        if NWBHDF5IO is None:
            raise ImportError("pynwb is not installed. Please `pip install pynwb` to read .nwb files.")
        file_name = os.path.basename(file_path)

        try:
            with NWBHDF5IO(file_path, 'r') as io:
                nwbfile = io.read()
                
                if 'responses' in nwbfile.analysis and 'ap_time' in nwbfile.analysis['responses']:
                    global_ap_times = nwbfile.analysis['responses']['ap_time'][:]
                else:
                    log.warning(f"No global ap_time found in {file_path}, skipping.")
                    return

                ophys_keys = [k for k in nwbfile.processing.keys() if 'ophys' in k]
                
                for ophys_key in ophys_keys:
                    ophys_module = nwbfile.processing[ophys_key]
                    
                    if 'Fluorescence' not in ophys_module.data_interfaces:
                        continue
                        
                    roi_series = ophys_module.data_interfaces['Fluorescence'].roi_response_series.get('RoiResponseSeries')
                    if roi_series is None:
                        continue
                        
                    fluo_times = roi_series.timestamps[:]
                    fluo_data = roi_series.data[:]
                    
                    if len(fluo_data.shape) > 1:
                        if fluo_data.shape[1] == len(fluo_times):
                            fluo_data = fluo_data.T  
                        
                        traces_mean = fluo_data[:, 0]
                    else:
                        traces_mean = fluo_data
                    
                    start_time = fluo_times[0]
                    end_time = fluo_times[-1]
                    mask = (global_ap_times >= start_time) & (global_ap_times <= end_time)
                    movie_spikes_time = global_ap_times[mask]

                    safe_ophys_key = ophys_key.replace(" ", "_")
                    trial_id = f"{file_name}_{safe_ophys_key}"

                    self._process_signal_to_windows(
                        traces_mean=traces_mean, 
                        fluo_times=fluo_times, 
                        events=movie_spikes_time, 
                        current_ephys_fs=1.0,       
                        trial_id=trial_id,
                        folder_name=folder_name
                    )
                    
        except Exception as e:
            log.warning(f"Failed parsing NWB components in {os.path.basename(file_path)}: {e}")


    def _process_signal_to_windows(self, traces_mean, fluo_times, events, current_ephys_fs, trial_id, folder_name=""):
        ""




        min_len = min(len(traces_mean), len(fluo_times))
        traces_mean = traces_mean[:min_len]
        fluo_times = fluo_times[:min_len]
        
        valid_mask = ~np.isnan(fluo_times) & ~np.isnan(traces_mean)
        traces_mean = traces_mean[valid_mask]
        fluo_times = fluo_times[valid_mask]
        
        if len(fluo_times) < self.segment_len: 
            return

        orig_frame_rate = 1 / np.nanmean(np.diff(fluo_times))

        # Denoising
        if self.denoise_method != "none":
            traces_mean = denoise_signal(
                traces_mean, method=self.denoise_method, params=self.denoise_params
            )

        # Resampling (downsample-only: never upsample low-rate data)
        if self.resample and orig_frame_rate > self.target_sampling_rate * 1.05:
            current_rate = self.target_sampling_rate
            traces_mean, fluo_times, _ = resample_trace(
                traces_mean, fluo_times, self.target_sampling_rate, orig_frame_rate
            )
        else:
            current_rate = orig_frame_rate

        # Spike Binning
        bin_step = 1 / current_rate
        bin_centers = np.linspace(fluo_times[0], fluo_times[-1], len(traces_mean))
        bin_edges = np.append(bin_centers, bin_centers[-1] + bin_step) - bin_step/2
        
        events_binned, _ = np.histogram(events / current_ephys_fs, bins=bin_edges)

        # Z-score normalization per trial (data-format agnostic)
        std_trace = np.std(traces_mean)
        if std_trace > 1e-8:
            traces_mean = (traces_mean - np.mean(traces_mean)) / std_trace

        # --- Spike Smoothing ---
        smoothing_samples = int(self.smoothing * current_rate)
        spike_rate = smooth_spikes(
                events_binned, smoothing=smoothing_samples, causal_kernel=self.causal_kernel)

        # --- Target Clipping (for indicators with targets > 1, e.g. OGB1 interneurons) ---
        if self.target_clip is not None:
            spike_rate = np.clip(spike_rate, 0.0, self.target_clip)

        # Final Cutting
        final_len = min(len(traces_mean), len(spike_rate))
        if final_len < self.segment_total_len:
            return
        
        traces_mean = traces_mean[:final_len]
        spike_rate = spike_rate[:final_len]
        
        # Sliding Window Generation
        start_positions = np.arange(0, final_len - self.segment_total_len + 1, step=self.stride).tolist()
        last_possible_start = final_len - self.segment_total_len

        if len(start_positions) == 0 or start_positions[-1] != last_possible_start:
            start_positions.append(last_possible_start)
        
        for start in start_positions:
            end = start + self.segment_total_len
            trace_window = traces_mean[start:end]
            rate_window = spike_rate[start:end]
            
            if np.any(np.isnan(trace_window)) or np.any(np.isnan(rate_window)):
                continue

            meta = {
                "trial_id": trial_id,
                "start_idx": start,                 
                "trial_len": final_len,
                "sampling_rate": current_rate,
                "indicator_id": parse_indicator_id(folder_name),
                "species_id": parse_species_id(folder_name),
                "tissue_id": parse_tissue_id(folder_name),
                "cell_type_id": parse_cell_type_id(folder_name),
            }
            
            self.data.append((
                trace_window.astype(np.float32), 
                rate_window.astype(np.float32),
                meta,
            ))
