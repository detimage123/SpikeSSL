"""
Data loading and preprocessing pipeline for spike inference model.

Data storage hierarchy:
    root folder/
    └── *.mat files  # MATLAB binary data files
        └── CAttached  # Core struct containing spike/fluorescence data
            ├── fluo_mean: array  # dF/F fluorescence trace (no artificial noise added)
            ├── fluo_raw: array  # Raw fluorescence trace
            ├── fluo_time: array  # Time points of fluorescence recording (unit: seconds)
            └── events_AP: array  # Timestamps of spike events (unit: samples, relative to ephys sampling rate)

Key definitions:
    - dF/F: Normalized fluorescence change (standard metric for calcium imaging)
    - ephys: Electrophysiology (the sampling rate of electrophysiological recording)
"""


import os
import glob
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from src.utils import pylogger
from src.utils.signal_processing import denoise_signal, smooth_spikes, resample_trace

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class CalciumDataset(Dataset):

    def __init__(
        self,
        data_dirs: List[str],
        sampling_rate: float,
        smoothing: float,
        segment_len: int,
        context_left: int = 30,
        context_right: int = 30,
        overleap_size: int = 2,
        resample: bool = True,
        omission_dict: Dict[str, List[int]] = None,
        causal_kernel: bool = False,
        denoise_method: str = "none",
        denoise_params: dict = None,
        feature_mode: str = "dual",
        ephys_fs: float = 1e4,
    ):
        
        super().__init__()


        self.data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.segment_len = segment_len
        self.segment_total_len = segment_len + context_left + context_right
        self.stride = self.segment_total_len // overleap_size

        self.target_sampling_rate = sampling_rate
        self.smoothing = smoothing
        
        self.resample = resample
        self.causal_kernel = causal_kernel
        self.denoise_method = denoise_method
        self.denoise_params = denoise_params if denoise_params is not None else {}
        self.ephys_fs = ephys_fs
        self.omission_dict = omission_dict if omission_dict is not None else {}
        valid_modes = ["raw", "diff", "dual"]
        if feature_mode not in valid_modes:
            raise ValueError(f"The feature_mode is: {feature_mode}. Must be one of {valid_modes}")
        self.feature_mode = feature_mode
        
        for folder in data_dirs:
            if os.path.exists(folder):
                self._load_folder(folder)
            else:
                log.warning(f"Folder not found: {folder}")



    def __getitem__(self, idx):
        trace_window, rate_window, fs = self.data[idx]
        
        y_final = rate_window.reshape(-1, 1)
          

        if self.feature_mode == "raw":
            x_output = trace_window.reshape(-1, 1)
            
        elif self.feature_mode == "diff":
            diff_feat = np.diff(trace_window, prepend=trace_window[0])
            x_output = diff_feat.reshape(-1, 1)

        elif self.feature_mode == "dual":
            diff_feat = np.diff(trace_window, prepend=trace_window[0])
            x_output = np.stack([trace_window, diff_feat], axis=1)

        
        return (
            torch.tensor(x_output, dtype=torch.float32), 
            torch.tensor(y_final, dtype=torch.float32),    
            torch.tensor(fs, dtype=torch.float32)
        )
    
    
    def __len__(self):
        return len(self.data)
    
    
    def _load_folder(self, folder: str):
        ""

        file_list = sorted(glob.glob(os.path.join(folder, '*_mini.mat')))
        
        folder_name = os.path.basename(folder)
        files_omission = self.omission_dict.get(folder_name, [])

        for index in sorted(files_omission, reverse=True):
            if index < len(file_list):
                del file_list[index]

        for file_path in file_list:
                
            try:    
                mat_data = sio.loadmat(file_path)['CAttached'][0]

                self._process_mat_data(mat_data)
    
            except Exception as e:
                log.warning(f"Failed to load {file_path}: {e}")
                continue


    def _process_mat_data(self, dataset_neuron_all):
        ""

        for i, trial in enumerate(dataset_neuron_all):
            try:
               
                keys = trial[0][0].dtype.descr
                keys_unfolded = list(sum(keys, ()))
                
                try:
                    traces_idx = int(keys_unfolded.index("fluo_mean") / 2)
                    time_idx = int(keys_unfolded.index("fluo_time") / 2)
                    event_idx = int(keys_unfolded.index("events_AP") / 2)
                except ValueError:
                    log.warning(f"Failed to found in trial {i}.")
                    continue

                events = trial[0][0][event_idx]
                events = events[~np.isnan(events)]
                fluo_times = np.squeeze(trial[0][0][time_idx])
                
                if len(fluo_times) < self.segment_len: 
                    continue
                orig_frame_rate = 1 / np.nanmean(np.diff(fluo_times))
                
                traces_mean = np.squeeze(trial[0][0][traces_idx])
                
                min_len = min(len(traces_mean), len(fluo_times))
                traces_mean = traces_mean[:min_len]
                fluo_times = fluo_times[:min_len]
                
                valid_mask = ~np.isnan(fluo_times) & ~np.isnan(traces_mean)
                traces_mean = traces_mean[valid_mask]
                fluo_times = fluo_times[valid_mask]
                
                if len(fluo_times) < self.segment_len: 
                    continue


                if self.denoise_method != "none":
                     traces_mean = denoise_signal(
                         traces_mean, 
                         method=self.denoise_method, 
                         params=self.denoise_params
                     )

                if self.resample:
                    current_rate = self.target_sampling_rate
                    if np.abs(self.target_sampling_rate - orig_frame_rate) / orig_frame_rate > 0.05:
                        traces_mean, fluo_times, _ = resample_trace(
                            traces_mean, fluo_times, self.target_sampling_rate, orig_frame_rate
                        )
                else:
                    current_rate = orig_frame_rate

                
                bin_step = 1 / current_rate
                
                bin_centers = np.linspace(fluo_times[0], fluo_times[-1], len(traces_mean))
                bin_edges = np.append(bin_centers, bin_centers[-1] + bin_step) - bin_step/2
                events_binned, _ = np.histogram(events / self.ephys_fs, bins=bin_edges)

                
                if self.resample:
                    spike_rate = smooth_spikes(
                        events_binned, 
                        self.smoothing, 
                        current_rate,
                        self.causal_kernel)
                    
                else:
                    smoothing_physical = self.smoothing / self.target_sampling_rate
                    smoothing_samples = int(smoothing_physical * current_rate)
                    smoothing_samples = max(1, smoothing_samples)
                    
                    spike_rate = smooth_spikes(
                        events_binned,
                        smoothing = smoothing_samples,
                        sampling_rate=current_rate,
                        causal_kernel=causal_kernel,
                        ).astype(np.float32, copy=False)

                final_len = min(len(traces_mean), len(spike_rate))
                if final_len < self.segment_total_len:
                    continue
                

                final_len = min(len(traces_mean), len(spike_rate))
                traces_mean = traces_mean[:final_len]
                spike_rate = spike_rate[:final_len]
                
                start_positions = np.arange(0, final_len - self.segment_total_len + 1, step=self.stride)
                
                for start in start_positions:
                    end = start + self.segment_total_len
                    
                    trace_window = traces_mean[start:end]
                    rate_window = spike_rate[start:end]
                    

                    if np.any(np.isnan(trace_window)) or np.any(np.isnan(rate_window)):
                        continue
                    

                    self.data.append((
                        trace_window.astype(np.float32), 
                        rate_window.astype(np.float32),
                        float(current_rate)
                    ))

            except Exception as e:
                if self.verbose > 1:
                    print(f"Error processing trial {i}: {e}")
                continue