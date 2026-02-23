# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 15:34:30 2026

@author: danie
"""

import sounddevice as sd
import soundfile as sf

def play_wav(path: str):
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr, blocking=False)
