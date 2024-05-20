import math
import numpy as np
import pandas as pd
import opensmile
from pyAudioAnalysis import audioBasicIO as aIO
import pyloudnorm as pyln


def load_audio(file):
    """Load audio from file path

    Parameters
    ----------
        filename (str)

    Returns
    -------
        sampling_freq (int): sampling frequence.
        signales (list[int]): time series giving the amplitude of each point.
    """
    if isinstance(file, str):
        file = file + ".wav"

    sampling_freq, signals = aIO.read_audio_file(file)
    return sampling_freq, signals


def compute_data_points(audio_signal, sampling_freq, win):
    """Function that returns a defined segmentation of the audio signal.
    The segmentation method is a rolling window of length 'win' without intersection.
    each data point belongs to a  time segment of the following form [win+t,win+t+1[

    Parameters
    ----------
    audio_signal : list[int]
    sampling_freq : int
    win : float

    Returns
    -------
    pd.DataFrame
        Three columns 'start', 'end' and 'signal' (list of data points.)
    """
    step = int((win / 2) * sampling_freq)
    boundaries = np.arange(0, len(audio_signal), step)
    segments = []
    for i in range(len(boundaries) - 1):
        min_, max_ = boundaries[i], boundaries[i + 1]
        segments.append(audio_signal[min_:max_])
    df_signals = pd.DataFrame(columns=["start", "end", "signal"])
    df_signals["signal"] = segments
    df_signals["start"] = [bound / sampling_freq for bound in boundaries[:-1]]
    df_signals["end"] = [bound / sampling_freq for bound in boundaries[1:]]
    return df_signals


def get_feature_engineering(df_signals, sampling_freq):
    """Function that generates audio features that are supposed
    to be higlhy predictive to spot laughters.

    Parameters
    ----------
    df_signals : pd.DataFrame
        giving for each audio segment the amplitude time series.
    """
    df_signals["id"] = df_signals[["end", "start"]].sum(axis=1).map(hash)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    segmented_features = []
    for segment in df_signals["signal"].to_list():
        res = smile.process_signal(segment, sampling_freq)
        segmented_features.append(res.values[0])
    features_names = res.columns
    method = "openSmile"
    print("Feature engineering with ", method, "ok")
    input_x = pd.DataFrame(segmented_features, columns=features_names)
    input_x["id"] = df_signals["id"]
    return input_x


def preprocess_audio(signals, sampling_freq):
    """Preprocessing the signals to prepare it for the inference by the ML models.

    Parameters
    ----------
    signals : list[int]
    sampling_freq : int

    Returns
    -------
    X : pd.DataFrame
        For each segment all the explicative features necessary for running the prediction model.
    df_signals : pd.DataFrame
        For each segment we have the starting and ending time and the series of amplitude.
    """
    win = 0.5  # time window of 0.5 sec.
    df_signals = compute_data_points(signals, sampling_freq, win)
    input_x = get_feature_engineering(df_signals, sampling_freq)
    return input_x, df_signals


def load_and_preprocess(filename):
    """Load and preprocess the audio"""
    print("reading file..")
    sampling_freq, signals = load_audio(filename)
    print("preprocessing audio..")
    input_x, df_signals = preprocess_audio(signals, sampling_freq)
    return input_x, df_signals


def get_signal_features(df_signals, sampling_freq):
    """

    Parameters
    ----------
    df_signals : _type_
        _description_
    """
    df_signals["duration"] = df_signals["end"] - df_signals["start"]
    df_signals["block_size"] = df_signals["duration"] * sampling_freq
    min_block_size = df_signals["block_size"].min() / sampling_freq
    block_size = math.floor(min_block_size * 100) / 100.0
    meter = pyln.Meter(sampling_freq)
    meter.block_size = block_size
    df_signals["loudness"] = df_signals["signal"].apply(lambda x: meter.integrated_loudness(x))
    df_signals["id"] = df_signals[["end", "start"]].sum(axis=1).map(hash)
    df_signals.set_index("id", inplace=True)
    df_signals.sort_values("start", inplace=True)
    return df_signals
