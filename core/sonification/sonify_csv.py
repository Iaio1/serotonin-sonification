# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:23:04 2026

@author: danie
"""

import os
import glob
import csv
import struct
import wave
import numpy as np

PITCH_BINS_MIDI = [48, 52, 55, 59, 60, 62, 64, 67, 71, 74]
DURATION_BINS_BEATS = [0.5, 1.0, 2.0, 4.0]
TEMPO_BPM = 120
BETWEEN_FILE_GAP_S = 0.5
INTERPEAK_TIME_SCALE = 0.125
AMPLITUDE_PITCH_LOW_PCT = 10
AMPLITUDE_PITCH_HIGH_PCT = 95
AMPLITUDE_PITCH_GAMMA = 1.4
AUC_DURATION_LOW_PCT = 5
AUC_DURATION_HIGH_PCT = 95
AUC_DURATION_GAMMA = 0.6
NOTE_DURATION_SCALE = 1.75
MIDI_TICKS_PER_BEAT = 480
MIDI_PROGRAM = 0
MIDI_VELOCITY = 90

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

def _midi_to_hz(midi_note):
    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))

def _normalize_value(x, lo, hi):
    if hi <= lo:
        return 0.5
    t = (float(x) - float(lo)) / (float(hi) - float(lo))
    return max(0.0, min(1.0, t))

def _snap_normalized_to_bins(norm_value, bins):
    if not bins:
        raise ValueError("bins must be a non-empty sequence")
    if len(bins) == 1:
        return bins[0]

    positions = np.linspace(0.0, 1.0, len(bins))
    idx = int(np.argmin(np.abs(positions - norm_value)))
    return bins[idx]

def _encode_var_len(value):
    value = int(max(0, value))
    buffer = value & 0x7F
    while True:
        value >>= 7
        if value == 0:
            break
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)

    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)

def _seconds_to_midi_ticks(seconds):
    ticks_per_second = (float(TEMPO_BPM) / 60.0) * float(MIDI_TICKS_PER_BEAT)
    return int(round(float(seconds) * ticks_per_second))

def _rows_to_note_events(rows, amp_col, auc_col, time_col, file_number_col, file_name_col, peak_number_col):
    amps = []
    aucs = []
    parsed_rows = []

    for r in rows:
        amp = _to_float(r.get(amp_col))
        auc = _to_float(r.get(auc_col))
        onset_s = _to_float(r.get(time_col)) if time_col else None
        if amp is None or auc is None:
            continue

        parsed_rows.append({
            "file_number": r.get(file_number_col) if file_number_col else None,
            "file_name": r.get(file_name_col) if file_name_col else None,
            "peak_number": r.get(peak_number_col) if peak_number_col else None,
            "source_amplitude": abs(amp),
            "source_auc": abs(auc),
            "source_peak_time_s": onset_s,
        })
        amps.append(abs(amp))
        aucs.append(abs(auc))

    if not parsed_rows:
        raise ValueError("No usable peak rows (amp/auc could not be parsed).")

    amp_lo, amp_hi = np.percentile(
        np.asarray(amps, dtype=float),
        [AMPLITUDE_PITCH_LOW_PCT, AMPLITUDE_PITCH_HIGH_PCT],
    )
    amp_lo = float(amp_lo)
    amp_hi = float(amp_hi)
    auc_lo, auc_hi = np.percentile(
        np.asarray(aucs, dtype=float),
        [AUC_DURATION_LOW_PCT, AUC_DURATION_HIGH_PCT],
    )
    auc_lo = float(auc_lo)
    auc_hi = float(auc_hi)
    seconds_per_beat = 60.0 / float(TEMPO_BPM)

    for row in parsed_rows:
        file_number = _to_float(row["file_number"])
        peak_number = _to_float(row["peak_number"])
        row["_file_number_sort"] = int(file_number) if file_number is not None else None
        row["_peak_number_sort"] = int(peak_number) if peak_number is not None else None
        row["_peak_time_sort"] = (
            float(row["source_peak_time_s"])
            if row["source_peak_time_s"] is not None else float("inf")
        )

    parsed_rows.sort(
        key=lambda row: (
            row["_file_number_sort"] if row["_file_number_sort"] is not None else float("inf"),
            row["file_name"] or "",
            row["_peak_time_sort"],
            row["_peak_number_sort"] if row["_peak_number_sort"] is not None else float("inf"),
        )
    )

    note_events = []
    running_onset_s = 0.0
    prev_file_key = None
    prev_peak_time_s = None

    for idx, row in enumerate(parsed_rows):
        file_key = (row["_file_number_sort"], row["file_name"])
        peak_time_s = row["source_peak_time_s"]

        if idx == 0:
            onset_s = 0.0
        elif file_key == prev_file_key:
            if peak_time_s is not None and prev_peak_time_s is not None:
                running_onset_s += (
                    max(0.0, float(peak_time_s) - float(prev_peak_time_s))
                    * INTERPEAK_TIME_SCALE
                )
            else:
                running_onset_s += 0.2
            onset_s = running_onset_s
        else:
            running_onset_s += BETWEEN_FILE_GAP_S
            onset_s = running_onset_s

        clipped_amplitude = min(max(row["source_amplitude"], amp_lo), amp_hi)
        normalized_amplitude = _normalize_value(clipped_amplitude, amp_lo, amp_hi)
        curved_pitch_value = normalized_amplitude ** AMPLITUDE_PITCH_GAMMA

        clipped_auc = min(max(row["source_auc"], auc_lo), auc_hi)
        normalized_auc = _normalize_value(clipped_auc, auc_lo, auc_hi)
        curved_duration_value = normalized_auc ** AUC_DURATION_GAMMA

        pitch_midi = int(_snap_normalized_to_bins(curved_pitch_value, PITCH_BINS_MIDI))
        duration_beats = float(_snap_normalized_to_bins(curved_duration_value, DURATION_BINS_BEATS))
        duration_s = duration_beats * seconds_per_beat
        rendered_duration_s = duration_s * NOTE_DURATION_SCALE

        file_number = _to_float(row["file_number"])
        peak_number = _to_float(row["peak_number"])

        note_events.append({
            "file_number": int(file_number) if file_number is not None else None,
            "file_name": row["file_name"],
            "peak_number": int(peak_number) if peak_number is not None else None,
            "onset_s": float(onset_s),
            "pitch_midi": pitch_midi,
            "duration_beats": duration_beats,
            "duration_s": rendered_duration_s,
            "source_amplitude": float(row["source_amplitude"]),
            "source_auc": float(row["source_auc"]),
            "source_peak_time_s": (
                float(row["source_peak_time_s"])
                if row["source_peak_time_s"] is not None else None
            ),
        })

        prev_file_key = file_key
        prev_peak_time_s = peak_time_s

    return note_events

def events_to_midi(note_events, midi_path):
    """Write the current quantized note events to a type-0 MIDI file."""
    if not midi_path:
        raise ValueError("midi_path must be a non-empty path")

    os.makedirs(os.path.dirname(midi_path) or ".", exist_ok=True)

    microseconds_per_beat = int(round(60_000_000 / float(TEMPO_BPM)))
    track_events = [
        (0, 0, b"\xFF\x51\x03" + microseconds_per_beat.to_bytes(3, "big")),
        (0, 1, bytes([0xC0, int(MIDI_PROGRAM) & 0x7F])),
    ]

    for note in note_events:
        pitch = int(note["pitch_midi"]) & 0x7F
        start_tick = max(0, _seconds_to_midi_ticks(note["onset_s"]))
        duration_tick = max(1, _seconds_to_midi_ticks(note["duration_s"]))
        end_tick = start_tick + duration_tick

        track_events.append((start_tick, 3, bytes([0x90, pitch, int(MIDI_VELOCITY) & 0x7F])))
        track_events.append((end_tick, 2, bytes([0x80, pitch, 0])))

    track_events.sort(key=lambda item: (item[0], item[1], item[2]))

    track_data = bytearray()
    last_tick = 0
    for abs_tick, _, message in track_events:
        delta = abs_tick - last_tick
        track_data.extend(_encode_var_len(delta))
        track_data.extend(message)
        last_tick = abs_tick

    track_data.extend(_encode_var_len(0))
    track_data.extend(b"\xFF\x2F\x00")

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, int(MIDI_TICKS_PER_BEAT))
    track = b"MTrk" + struct.pack(">I", len(track_data)) + bytes(track_data)

    with open(midi_path, "wb") as fh:
        fh.write(header)
        fh.write(track)

    return midi_path

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
        peak_number_col = _find_col(reader.fieldnames, ["Peak Number", "peak_number"])
        file_number_col = _find_col(reader.fieldnames, ["File Number", "file_number"])
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

    note_events = _rows_to_note_events(
        rows,
        amp_col=amp_col,
        auc_col=auc_col,
        time_col=time_col,
        file_number_col=file_number_col,
        file_name_col=file_col,
        peak_number_col=peak_number_col,
    )

    SR = 44100
    total_len = float(max(ev["onset_s"] + ev["duration_s"] for ev in note_events) + 0.25)
    audio = np.zeros(int(total_len * SR), dtype=np.float32)

    for event in note_events:
        freq_hz = _midi_to_hz(event["pitch_midi"])
        note = _sine(freq_hz, event["duration_s"], SR) * 0.5
        i0 = int(round(event["onset_s"] * SR))
        i1 = min(audio.size, i0 + note.size)
        if i0 < audio.size:
            audio[i0:i1] += note[: (i1 - i0)]

    out_wav = os.path.join(os.path.dirname(csv_path) or ".", "sonification.wav")
    _write_wav(out_wav, audio, SR)
    out_midi = os.path.join(os.path.dirname(csv_path) or ".", "sonification.mid")
    events_to_midi(note_events, out_midi)
    return out_wav

def sonify_from_folder(folder: str, replicate: str = None, file_name: str = None) -> str:
    """Backward-compatible wrapper that resolves the latest CSV within a folder."""
    csv_path = _latest_csv_in_folder(folder)
    return sonify_from_csv(csv_path, replicate=replicate, file_name=file_name)
