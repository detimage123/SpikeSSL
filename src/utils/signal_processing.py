import numpy as np
import scipy.signal as signal
from scipy.signal import wiener, detrend, convolve
from typing import Tuple, Optional, Dict


def denoise_signal(
    trace: np.ndarray, 
    method: str = "wiener", 
    params: Optional[Dict] = None
) -> np.ndarray:
    ""


    if params is None:
        params = {}
        
    if method == "wiener":

        win_size = params.get("win_size", 15)
        if win_size % 2 == 0: win_size += 1
        
        signal_detrended = detrend(trace)
        trend = trace - signal_detrended
        
        try:
            
            fwd = wiener(signal_detrended, mysize=win_size)
            bwd = wiener(signal_detrended[::-1], mysize=win_size)[::-1]
            return (fwd + bwd) / 2 + trend
        except Exception:
            return trace
        
    if method == "median":
        kernel_size = params.get("kernel_size", 3)
        if kernel_size % 2 == 0: kernel_size += 1
        return signal.medfilt(trace, kernel_size=kernel_size)
    
    if method == "savgol":
        win_size = params.get("win_size", 15)
        polyorder = params.get("polyorder", 3)
        if win_size % 2 == 0: win_size += 1
        return signal.savgol_filter(trace, window_length=win_size, polyorder=polyorder)
            
    elif method in {
        "none",
        "selfsup_masked_ae",
        "selfsup_noise2void_1d",
        "selfsup_blindspot_tcn",
        "selfsup_contrastive_denoise",
    }:
        # selfsup_* methods are intended for model-side joint optimization.
        # Keep raw trace here to avoid conflicting offline denoising.
        return trace
    else:
        raise ValueError(f"Unknown denoise method: {method}")


def resample_trace(
    trace: np.ndarray, 
    times: np.ndarray, 
    target_fs: float, 
    orig_fs: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    ""
    duration = len(trace) / orig_fs
    new_num_samples = int(duration * target_fs)
    
    new_trace = signal.resample(trace, new_num_samples)
    new_times = np.linspace(times[0], times[-1], new_num_samples)
    
    return new_trace, new_times, target_fs


def smooth_spikes(
    binned_spikes: np.ndarray, 
    smoothing: float, 
    causal_kernel: bool = False
    
) -> np.ndarray:
    ""



    
    sigma = max(1.0, smoothing)
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0: kernel_size += 1
    
    kernel = signal.windows.gaussian(kernel_size, std=sigma)
    kernel /= kernel.sum()
    
    if causal_kernel:
        kernel_causal = kernel[kernel_size//2:] 
        kernel_causal /= kernel_causal.sum()
        return convolve(binned_spikes, kernel_causal, mode='full')[:len(binned_spikes)]
    else:
        return convolve(binned_spikes, kernel, mode='same')