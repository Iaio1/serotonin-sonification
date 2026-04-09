# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:23:04 2026

@author: danie
"""

import os
import glob
import csv
import wave
import numpy as np

def _latest_csv_in_folder(folder: str) -> str:
    patterns = [
        os.path.join(folder, "peak_characteristics*.csv"),
        os.path.join(folder, "**", "peak_characteristics*.csv"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No peak_characteristics*.csv found in:\n{folder}\n\n"
            "Run 'Characteristic Extraction' first, or choose the correct folder."
        )
    return max(matches, key=os.path.getmtime)

def _find_col(fieldnames, candidates):
    low = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=float), 0.0, 1.0
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return None, lo, hi
    return None, lo, hi

def _exp_map(x, lo, hi, out_lo, out_hi):
    # exponential mapping (better for pitch)
    if hi <= lo:
        return (out_lo + out_hi) / 2.0
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return out_lo * ((out_hi / out_lo) ** t)

def _lin_map(x, lo, hi, out_lo, out_hi):
    if hi <= lo:
        return (out_lo + out_hi) / 2.0
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)

def _sine(freq_hz, dur_s, sr):
    n = max(1, int(round(dur_s * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    y = np.sin(2 * np.pi * float(freq_hz) * t)

    # fade in/out to avoid clicks
    a = int(0.01 * sr)
    r = int(0.02 * sr)
    env = np.ones(n, dtype=np.float32)
    if a > 1:
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if r > 1 and r < n:
        env[-r:] = np.linspace(1, 0, r, dtype=np.float32)

    return y * env

def _write_wav(path, audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-9:
        audio = 0.95 * audio / peak
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def sonify_from_csv(csv_path: str, replicate: str = None, file_name: str = None) -> str:
    """
    Convert an exported spontaneous peak-characteristics CSV to sound:
      amplitude -> pitch, auc -> duration
    and peak time -> note onset. Returns the generated WAV path.
    """
    if not csv_path:
        raise ValueError("csv_path must be a non-empty path")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Peak characteristics CSV not found:\n{csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        amp_col = _find_col(reader.fieldnames, ["Amplitude", "Amp", "amp"])
        auc_col = _find_col(reader.fieldnames, ["AUC", "auc", "Area"])
        time_col = _find_col(reader.fieldnames, ["Peak Time (s)", "Peak Time", "peak_s", "peak time"])
        rep_col  = _find_col(reader.fieldnames, ["Replicate", "replicate"])
        file_col = _find_col(reader.fieldnames, ["File Name", "File", "Filename", "file"])

        if amp_col is None or auc_col is None:
            raise ValueError(f"CSV missing required columns. Found: {reader.fieldnames}")

        rows = list(reader)

    def _norm(s: str) -> str:
        return (s or "").strip().lower()
    
    def _norm_file(s: str) -> str:
        s = _norm(s)
        s = os.path.basename(s)              # drop any folder
        s = s.replace("\\", "/")
        return s
    
    def _stem(s: str) -> str:
        s = _norm_file(s)
        return os.path.splitext(s)[0]        # remove extension
    
    rows_all = rows[:]  # keep original, in case filtering fails
    
    # Filter replicate (case-insensitive)
    if replicate is not None and rep_col is not None:
        rep_target = _norm(replicate)
        rows = [r for r in rows if _norm(r.get(rep_col, "")) == rep_target]
    
    # Filter file (basename + case-insensitive + allow "stem" match)
    if file_name is not None and file_col is not None:
        file_target = _norm_file(file_name)
        file_target_stem = _stem(file_name)
    
        def file_matches(val: str) -> bool:
            v = _norm_file(val)
            return (v == file_target) or (_stem(v) == file_target_stem)
    
        rows = [r for r in rows if file_matches(r.get(file_col, ""))]
    
    # If filtering produced nothing, fall back to *no filtering* (still sonifies something)
    if not rows:
        rows = rows_all


    amps = []
    aucs = []
    times = []

    for r in rows:
        a = _to_float(r.get(amp_col))
        u = _to_float(r.get(auc_col))
        t = _to_float(r.get(time_col)) if time_col else None
        if a is None or u is None:
            continue
        amps.append(abs(a))
        aucs.append(abs(u))
        times.append(t if t is not None else len(times) * 0.2)  # fallback spacing

    if len(amps) == 0:
        raise ValueError("No usable peak rows (amp/auc could not be parsed).")

    amps = np.array(amps, dtype=float)
    aucs = np.array(aucs, dtype=float)
    times = np.array(times, dtype=float)

    order = np.argsort(times)
    amps, aucs, times = amps[order], aucs[order], times[order]

    # Auto-scale to ANY CSV (no fixed dataset values)
    a_lo, a_hi = float(np.min(amps)), float(np.max(amps))
    u_lo, u_hi = float(np.min(aucs)), float(np.max(aucs))

    # Mapping ranges (tweak later if you want)
    FMIN, FMAX = 220.0, 880.0     # pitch range
    DMIN, DMAX = 0.08, 0.60       # duration range
    SR = 44100

    freqs = np.array([_exp_map(x, a_lo, a_hi, FMIN, FMAX) for x in amps], dtype=float)
    durs  = np.array([_lin_map(x, u_lo, u_hi, DMIN, DMAX) for x in aucs], dtype=float)

    total_len = float(np.max(times + durs) + 0.25)
    audio = np.zeros(int(total_len * SR), dtype=np.float32)

    for f, d, t0 in zip(freqs, durs, times):
        note = _sine(f, d, SR) * 0.5
        i0 = int(round(t0 * SR))
        i1 = min(audio.size, i0 + note.size)
        if i0 < audio.size:
            audio[i0:i1] += note[: (i1 - i0)]

    out_wav = os.path.join(os.path.dirname(csv_path) or ".", "sonification.wav")
    _write_wav(out_wav, audio, SR)
    return out_wav

def sonify_from_folder(folder: str, replicate: str = None, file_name: str = None) -> str:
    """Backward-compatible wrapper that resolves the latest CSV within a folder."""
    csv_path = _latest_csv_in_folder(folder)
    return sonify_from_csv(csv_path, replicate=replicate, file_name=file_name)
