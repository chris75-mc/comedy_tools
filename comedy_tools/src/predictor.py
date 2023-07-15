"""Predictor class"""
from os import path
import pandas as pd
import numpy as np
from abc import ABC
from .audio_processing import load_audio, compute_data_points, get_feature_engineering
from .artifacts.Enums import ConfigEnums
from .helpers.utils import load_model


class Predictor(ABC):
    """Predictor"""

    def __init__(self, filename: str = "", audio_proprieties: list = []):
        """Init function"""
        self.sampling_win = ConfigEnums.SEGMENTATION_PARAMS.value.get("sampling_window", 0.5)
        self.segmented_signal, self.input_x = pd.DataFrame(), pd.DataFrame()
        if filename != "":
            self.sampling_freq, self.signal = load_audio(filename)
        elif len(audio_proprieties) > 0:
            self.sampling_freq, self.signal = audio_proprieties[0], np.array(audio_proprieties[1])
        dir_path, _ = path.split(__file__)
        model_path = path.join(dir_path, "artifacts", "models", "catboost_2.cbm")
        self.model = load_model(model_path, type="catboost")

    def preprocess_signal(self):
        """preprocess_signal"""
        self.segmented_signal = compute_data_points(self.signal, self.sampling_freq, self.sampling_win)

    def get_feature_engineering(
        self,
    ):
        """get_feature_engineering"""
        self.input_x = get_feature_engineering(self.segmented_signal, self.sampling_freq)

    def predict(
        self,
    ):
        """predict"""
        self.preprocess_signal()
        self.get_feature_engineering()
        predictions = self.model.predict_proba(self.input_x)[:, 1]
        return predictions

    def get_class(self):
        predictions = self.predict()
        thresh = ConfigEnums.PROB_THRESHOLD.value
        classes = ["Laugh" if pred > thresh else "Speech" for pred in predictions]
        self.segmented_signal.loc[:, "class"] = classes
