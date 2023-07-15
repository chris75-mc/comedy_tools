"""Features to qualifiate an audio signal from a laughter point of view"""
from time import gmtime
from time import strftime
import numpy as np
import pandas as pd
from datetime import timedelta
from humanfriendly import format_timespan


def get_strike_dict(time_btw_laughter):
    dict_dense_laugh = {}
    cursor = 0
    durations = []
    temp = []
    for (index, duration) in enumerate(time_btw_laughter):
        if duration <= 15:
            durations.append(duration)
            temp.append(index)
        else:
            if len(temp) > 0:
                cursor = (temp[0]) // 2

            dict_dense_laugh[cursor] = durations
            temp = []
            durations = []
    if len(temp) > 0:
        cursor = temp[0] // 2
    dict_dense_laugh[cursor] = durations
    return dict_dense_laugh


def spot_sucessive_strikes(df_laugh, max_delay=10):
    df_temp = pd.DataFrame()
    df_temp["time"] = df_laugh["start"].to_list() + df_laugh["end"].to_list()
    df_temp.sort_values("time", inplace=True)
    df_temp.reset_index(drop=True, inplace=True)
    time_btw_laughter = df_temp["time"].diff().dropna()
    dict_dense_laugh = get_strike_dict(time_btw_laughter)
    return dict_dense_laugh


def get_kpis(df_signals):
    res = {}
    duration = df_signals["duration"].sum()
    res["duration"] = timedelta(seconds=duration)

    filter_laugh = df_signals["class"] == "Laugh"

    res["total_laugh_duration"] = timedelta(seconds=df_signals[filter_laugh]["duration"].sum())
    res["laugh_importance"] = str(round(res["total_laugh_duration"] / res["duration"], 2) * 100) + "%"
    res["n_laugh"] = np.sum(filter_laugh)
    n_mins = strftime("%M", gmtime(duration))
    res["laugh_per_min"] = str(round(res["n_laugh"] / int(n_mins), 2)) + " laugh/pm"

    # mean duration of a laugh
    res["mean_laugh_duration"] = df_signals[filter_laugh]["duration"].mean()
    # best laugh
    longest_laugh_index = df_signals[filter_laugh]["duration"].idxmax()
    res["longest_laugh_duration"] = timedelta(seconds=df_signals.loc[longest_laugh_index, "duration"])

    index_max = df_signals[filter_laugh]["loudness"].idxmax()
    res["time_loudest_laugh"] = [
        format_timespan(time) for time in df_signals.loc[index_max, ["start", "end"]].to_list()
    ]
    res["loudest_laugh"] = str(round(df_signals.loc[index_max, "loudness"], 2)) + " dB"

    # time before first laugh
    res["time_before_first_laugh"] = df_signals[filter_laugh]["start"].values[0]

    # best successive strike of laugh
    dict_dense_laugh = spot_sucessive_strikes(df_signals[filter_laugh])
    laugh_index = max(dict_dense_laugh.items(), key=lambda x: len(x[1]))[0]
    res["longest_laugh_serie"] = len(dict_dense_laugh[laugh_index])
    laugh_id = df_signals[filter_laugh].iloc[laugh_index].name
    previous_speech_index = np.where(df_signals.index == laugh_id)[0][0] - 1
    if previous_speech_index >= 0:
        res["most_hilarous_speech"] = [
            format_timespan(time) for time in df_signals.iloc[previous_speech_index][["start", "end"]].to_list()
        ]
    else:
        res["most_hilarous_speech"] = "First seconds"

    feat_convert_to_str = [
        "duration",
        "total_laugh_duration",
        "mean_laugh_duration",
        "longest_laugh_duration",
        "time_before_first_laugh",
    ]
    for feat in feat_convert_to_str:
        res[feat] = format_timespan(res[feat])

    df_kpis = pd.Series(res)
    return df_kpis
