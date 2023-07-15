"""Useful functions"""
import json
from catboost import CatBoostClassifier


def load_json(path):
    """Load a json file giving its path"""
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file


def load_model(model_path, type="catboost"):
    if type == "catboost":
        model = CatBoostClassifier().load_model(model_path, format="cbm")
        return model
    return None
