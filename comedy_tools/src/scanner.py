"""Scan a tagged signal"""
from os import path
import pandas as pd
from abc import ABC
from .audio_processing import get_signal_features
from .kpis import get_kpis


class Scanner(ABC):
    """Implementation of the audio scanning block."""

    def __init__(self, df_signals: pd.DataFrame, sampling_freq: float):
        self.segmented_signal = df_signals
        self.sampling_freq = sampling_freq

    def compute_sound_features(self):
        self.segmented_signal = get_signal_features(self.segmented_signal, self.sampling_freq)

    def scan(self):
        self.compute_sound_features()
        return get_kpis(self.segmented_signal)
