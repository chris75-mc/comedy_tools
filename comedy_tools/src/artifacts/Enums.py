""" This module contains enums for prediction"""
from enum import Enum


class ConfigEnums(Enum):
    """Enum"""

    SEGMENTATION_PARAMS = {"sampling_window": 0.5}
    PROB_THRESHOLD = 0.2
