from .base import Processor
import numpy as np
from scipy.signal import find_peaks
from PyQt5.QtCore import QSettings

class FindAmplitudeMultiple(Processor):
    def __init__(self, peak_position, min_prominence=0.05, min_decay_ratio=0.1, 
                 rise_window_sec=1.0, decay_window_sec=20.0, max_peaks=10):
        """
        Initialize FindAmplitudeMultiple for spontaneous signal analysis.
        
        Args:
            peak_position: Column index for peak detection
            min_prominence: Minimum peak prominence (lowered for spontaneous)
            min_decay_ratio: Minimum decay ratio from peak
            rise_window_sec: Time window in seconds to check for rise (reduced for spontaneous)
            decay_window_sec: Time window in seconds to check for decay (reduced for spontaneous)
            max_peaks: Maximum number of peaks to return
        """
        self.peak_position = peak_position
        self.min_prominence = min_prominence
        self.min_decay_ratio = min_decay_ratio
        self.rise_window_sec = rise_window_sec
        self.decay_window_sec = decay_window_sec
        self.max_peaks = max_peaks

    def _find_adaptive_time_windows(self, fx, peak_idx, freq):
        """
        Dynamically find optimal rise and decay windows for spontaneous signals.
        Uses shorter time windows compared to stimulated signals.
        """
        # Adjusted minimum and maximum window constraints for spontaneous signals
        MIN_RISE_TIME_SEC = 0.2   # Reduced from 0.5s - spontaneous events can be faster
        MAX_RISE_TIME_SEC = 3.0   # Reduced from 5.0s
        MIN_DECAY_TIME_SEC = 1.0  # Reduced from 2.0s
        MAX_DECAY_TIME_SEC = 10.0 # Reduced from 20.0s

        # Convert to samples
        min_rise_samples = max(3, int(MIN_RISE_TIME_SEC * freq))
        max_rise_samples = int(MAX_RISE_TIME_SEC * freq)
        min_decay_samples = max(5, int(MIN_DECAY_TIME_SEC * freq))
        max_decay_samples = int(MAX_DECAY_TIME_SEC * freq)

        peak_val = fx[peak_idx]

        # Find adaptive rise window
        rise_window = self._find_rise_window(fx, peak_idx, min_rise_samples, max_rise_samples)

        # Find adaptive decay window
        decay_window = self._find_decay_window(fx, peak_idx, peak_val, min_decay_samples, max_decay_samples)

        # Convert back to seconds for reporting
        rise_time_sec = rise_window / freq
        decay_time_sec = decay_window / freq

        print(f"Adaptive windows for peak at {peak_idx}: rise={rise_time_sec:.1f}s, decay={decay_time_sec:.1f}s")

        return rise_window, decay_window, rise_time_sec, decay_time_sec

    def _find_rise_window(self, fx, peak_idx, min_samples, max_samples):
        """Find the optimal rise window for spontaneous signals."""
        peak_val = fx[peak_idx]
        start_search = max(0, peak_idx - max_samples)

        # Find baseline level
        early_points = fx[start_search:max(1, start_search + min_samples // 2)]
        if len(early_points) > 0:
            baseline = np.median(early_points)
        else:
            baseline = 0

        # More sensitive rise threshold for spontaneous signals
        rise_threshold = baseline + 0.05 * (peak_val - baseline)  # Reduced from 0.1

        # Find where signal first crosses the rise threshold
        rise_start_idx = peak_idx
        for i in range(peak_idx - 1, start_search - 1, -1):
            if fx[i] < rise_threshold:
                rise_start_idx = i
                break

        rise_window = max(min_samples, peak_idx - rise_start_idx)
        rise_window = min(max_samples, rise_window)

        return rise_window

    def _find_decay_window(self, fx, peak_idx, peak_val, min_samples, max_samples):
        """Find the optimal decay window for spontaneous signals."""
        end_search = min(len(fx), peak_idx + max_samples)

        if end_search <= peak_idx + min_samples:
            return min_samples

        # Find baseline
        far_start = min(len(fx) - 5, peak_idx + max_samples - 5)
        if far_start < len(fx):
            far_points = fx[far_start:len(fx)]
            if len(far_points) > 2:
                baseline = np.median(far_points)
            else:
                baseline = fx[-1] if len(fx) > 0 else 0
        else:
            baseline = fx[min(len(fx) - 1, peak_idx + min_samples)]

        # Adjusted decay thresholds for spontaneous signals
        decay_70_threshold = baseline + 0.7 * (peak_val - baseline)  # 70% instead of 90%
        decay_30_threshold = baseline + 0.3 * (peak_val - baseline)  # 30% instead of 50%

        # Find threshold crossings
        decay_70_idx = None
        decay_30_idx = None

        for i in range(peak_idx + 1, end_search):
            if i >= len(fx):
                break

            if decay_70_idx is None and fx[i] <= decay_70_threshold:
                decay_70_idx = i
            if decay_30_idx is None and fx[i] <= decay_30_threshold:
                decay_30_idx = i
                break

        # Determine decay window
        if decay_30_idx is not None:
            decay_window = min(max_samples, (decay_30_idx - peak_idx) + 5)
        elif decay_70_idx is not None:
            decay_window = min(max_samples, (decay_70_idx - peak_idx) * 2)
        else:
            decay_window = self._find_decay_by_slope(fx, peak_idx, min_samples, max_samples)

        decay_window = max(min_samples, decay_window)
        return decay_window

    def _find_decay_by_slope(self, fx, peak_idx, min_samples, max_samples):
        """Fallback: find decay window by slope analysis for spontaneous signals."""
        end_search = min(len(fx), peak_idx + max_samples)

        if end_search <= peak_idx + min_samples:
            return min_samples

        window_size = 3  # Smaller window for spontaneous signals
        slopes = []

        for i in range(peak_idx + window_size, end_search - window_size):
            if i + window_size >= len(fx):
                break
            slope = (fx[i + window_size] - fx[i - window_size]) / (2 * window_size)
            slopes.append((i, slope))

        if not slopes:
            return min_samples

        # More sensitive slope threshold for spontaneous signals
        slope_threshold = 0.002

        for i, slope in slopes:
            if abs(slope) < slope_threshold:
                decay_window = min(max_samples, i - peak_idx + 5)
                return max(min_samples, decay_window)

        return min(max_samples, min_samples * 2)

    def _validate_peak_with_adaptive_windows(self, fx, peak_idx, freq):
        """Validate peak using adaptive time windows for spontaneous signals."""
        rise_window, decay_window, rise_time_sec, decay_time_sec = self._find_adaptive_time_windows(fx, peak_idx, freq)

        decay_valid = self._validate_peak_decay_adaptive(fx, peak_idx, decay_window)
        rise_valid = self._validate_left_side_rise_adaptive(fx, peak_idx, rise_window)

        # More lenient window validation for spontaneous signals
        window_valid = (rise_time_sec >= 0.2 and decay_time_sec >= 1.0)

        return decay_valid and rise_valid and window_valid, {
            'rise_window_samples': rise_window,
            'decay_window_samples': decay_window,
            'rise_time_sec': rise_time_sec,
            'decay_time_sec': decay_time_sec
        }

    def _validate_peak_decay_adaptive(self, fx, peak_idx, decay_window):
        """Validate decay for spontaneous signals."""
        peak_val = fx[peak_idx]
        end_idx = min(len(fx), peak_idx + decay_window)

        if end_idx <= peak_idx + 3:  # Reduced minimum points
            return False

        decay_region = fx[peak_idx:end_idx]
        x_vals = np.arange(len(decay_region))
        
        if len(x_vals) < 3:
            return False

        try:
            coeffs = np.polyfit(x_vals, decay_region, 1)
            slope = coeffs[0]

            if slope >= 0:
                return False

            # More lenient decay requirement for spontaneous signals
            end_val = np.median(decay_region[-3:])  # Last 3 points
            decay_ratio = (peak_val - end_val) / peak_val if peak_val != 0 else 0

            return decay_ratio >= 0.1  # Reduced from 0.2

        except:
            return False

    def _validate_left_side_rise_adaptive(self, fx, peak_idx, rise_window):
        """Validate rise for spontaneous signals."""
        peak_val = fx[peak_idx]
        start_idx = max(0, peak_idx - rise_window)

        if peak_idx - start_idx < 3:  # Reduced minimum points
            return False

        rise_region = fx[start_idx:peak_idx + 1]
        x_vals = np.arange(len(rise_region))

        try:
            coeffs = np.polyfit(x_vals, rise_region, 1)
            slope = coeffs[0]

            if slope <= 0:
                return False

            # More lenient rise requirement
            start_val = np.median(rise_region[:3])  # First 3 points
            rise_amplitude = peak_val - start_val

            return rise_amplitude >= 0.03  # Reduced from 0.05

        except:
            return False

    def process(self, data, context=None):
            """
            Robust multi-peak detection on the I–T profile using:
            smoothing + drift removal + MAD noise + hysteresis segmentation.
            Stores per-peak dictionaries in context['spontaneous_peaks'].
            """
            from PyQt5.QtCore import QSettings
            import numpy as np

            print(">>> FindAmplitudeMultiple RUNNING", data.shape, "peak_position", self.peak_position)

            settings = QSettings("HashemiLab", "NeuroStemVolt")
            freq = settings.value("acquisition_frequency", 10, type=int)

            # 1) Extract I–T profile
            fx = np.asarray(data[:, self.peak_position], dtype=float)
            n = fx.size
            if n < 10:
                if context is not None:
                    context["spontaneous_peaks"] = []
                    context["num_peaks_detected"] = 0
                return data

            # ---- Tunable parameters (good defaults for your traces) ----
            smooth_sec = 0.7  # light smoothing
            baseline_sec = 15.0  # drift window
            k_enter = 3.5  # enter threshold = k_enter * sigma (3–4 is typical)
            k_exit = 2.0  # exit threshold  = k_exit  * sigma
            min_width_sec = 0.25
            min_gap_sec = 0.4
            polarity = "both"  # handles inverted peaks too
            # -----------------------------------------------------------

            # 2) Smooth (Savitzky–Golay if possible; fallback to moving average)
            try:
                from scipy.signal import savgol_filter
                w = max(5, int(round(smooth_sec * freq)))
                if w % 2 == 0:
                    w += 1
                w = min(w, n if n % 2 == 1 else n - 1)
                fx_s = savgol_filter(fx, window_length=w, polyorder=3, mode="interp")
            except Exception:
                w = max(3, int(round(smooth_sec * freq)))
                kernel = np.ones(w) / w
                fx_s = np.convolve(fx, kernel, mode="same")

            # 3) Remove drift baseline (rolling median; robust to steps)
            try:
                import pandas as pd
                bw = max(5, int(round(baseline_sec * freq)))
                if bw % 2 == 0:
                    bw += 1
                baseline = (
                    pd.Series(fx_s)
                    .rolling(window=bw, center=True, min_periods=1)
                    .median()
                    .to_numpy()
                )
            except Exception:
                # Fallback: median filter
                from scipy.signal import medfilt
                bw = max(5, int(round(baseline_sec * freq)))
                if bw % 2 == 0:
                    bw += 1
                baseline = medfilt(fx_s, kernel_size=bw)

            resid = fx_s - baseline

            # Polarity handling (some traces may be inverted)
            if polarity == "negative":
                resid_use = -resid
            elif polarity == "both":
                # pick the stronger side automatically
                resid_use = resid if np.nanmax(resid) >= abs(np.nanmin(resid)) else -resid
            else:
                resid_use = resid

            # 4) Robust noise sigma (MAD) estimated from the "quiet" part (avoids peaks inflating sigma)
            abs_r = np.abs(resid_use)
            quiet = abs_r <= np.quantile(abs_r, 0.7)  # lowest 70% magnitude = mostly noise
            r0 = resid_use[quiet] if np.any(quiet) else resid_use

            med = np.median(r0)
            mad = np.median(np.abs(r0 - med))
            sigma = 1.4826 * mad if mad > 1e-12 else (np.std(r0) + 1e-12)

            enter_th = k_enter * sigma
            exit_th = k_exit * sigma

            print(f">>> sigma={sigma:.4g} enter_th={enter_th:.4g} exit_th={exit_th:.4g}")

            min_width = max(1, int(round(min_width_sec * freq)))
            min_gap = max(1, int(round(min_gap_sec * freq)))

            # 5) Hysteresis segmentation
            peaks = []
            i = 0
            while i < n:
                if resid_use[i] >= enter_th:
                    # expand left to exit threshold
                    start = i
                    while start > 0 and resid_use[start - 1] > exit_th:
                        start -= 1
                    # expand right to exit threshold
                    end = i
                    while end < n - 1 and resid_use[end + 1] > exit_th:
                        end += 1

                    # enforce minimum width
                    if (end - start + 1) >= min_width:
                        seg = resid_use[start:end + 1]
                        peak_rel = int(np.argmax(seg))
                        peak_idx = start + peak_rel
                        amp = float(seg[peak_rel])

                        # basic features already useful for music mapping
                        # (AUC in "nA·s" if your axis is nA and x is seconds)
                        auc = float(np.trapz(seg, dx=1.0 / freq))

                        # FWHM (seconds)
                        half = 0.5 * amp
                        left = peak_idx
                        while left > start and resid_use[left] >= half:
                            left -= 1
                        right = peak_idx
                        while right < end and resid_use[right] >= half:
                            right += 1
                        fwhm_s = float((right - left) / freq)

                        peaks.append({
                            "start_idx": int(start),
                            "peak_idx": int(peak_idx),
                            "end_idx": int(end),
                            "start_s": float(start / freq),
                            "peak_s": float(peak_idx / freq),
                            "end_s": float(end / freq),
                            "amp": float(amp),  # baseline-removed amplitude
                            "auc": float(auc),
                            "fwhm_s": float(fwhm_s),
                            "rise_s": float((peak_idx - start) / freq),
                            "decay_s": float((end - peak_idx) / freq),
                            "snr": float(amp / (sigma + 1e-12)),
                        })

                    i = end + 1
                else:
                    i += 1

            # 6) Merge events that are too close (prevents double-counting in noisy traces)
            if peaks:
                merged = [peaks[0]]
                for p in peaks[1:]:
                    prev = merged[-1]
                    if p["start_idx"] - prev["end_idx"] <= min_gap:
                        # merge by extending end; keep the higher peak
                        prev["end_idx"] = max(prev["end_idx"], p["end_idx"])
                        prev["end_s"] = prev["end_idx"] / freq

                        if p["amp"] > prev["amp"]:
                            # replace peak location/features if new peak is larger
                            for k in ["peak_idx", "peak_s", "amp", "auc", "fwhm_s", "rise_s", "decay_s", "snr"]:
                                prev[k] = p[k]
                    else:
                        merged.append(p)
                peaks = merged

            print(">>> detected", len(peaks), "peaks at", [p["peak_idx"] for p in peaks])

            # store results for UI / export / later sonification
            if context is not None:
                context["spontaneous_peaks"] = peaks
                context["num_peaks_detected"] = len(peaks)
                context["peak_detection_params"] = {
                    "freq_hz": freq,
                    "smooth_sec": smooth_sec,
                    "baseline_sec": baseline_sec,
                    "k_enter": k_enter,
                    "k_exit": k_exit,
                    "min_width_sec": min_width_sec,
                    "min_gap_sec": min_gap_sec,
                    "polarity": polarity,
                    "sigma": float(sigma),
                }

                # Backwards compatibility with the rest of the app:
                if peaks:
                    # primary peak = max amplitude
                    best = max(peaks, key=lambda d: d["amp"])
                    context["primary_peak_position"] = best["peak_idx"]
                    context["primary_peak_value"] = best["amp"]
                    context["peak_amplitude_positions"] = [p["peak_idx"] for p in peaks]
                    context["peak_amplitude_values"] = [p["amp"] for p in peaks]
                else:
                    context["primary_peak_position"] = None
                    context["primary_peak_value"] = None
                    context["peak_amplitude_positions"] = []
                    context["peak_amplitude_values"] = []

            return data
