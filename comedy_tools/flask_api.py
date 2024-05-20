"""Flask api"""
import logging
import traceback
from flask import Flask, request

try:
    from .src.predictor import Predictor
    from .src.scanner import Scanner
except:
    from src.predictor import Predictor
    from src.scanner import Scanner

app = Flask(__name__)


@app.route("/")
def index():
    return "Index Page"


@app.route("/hello")
def hello():
    return {"Hello, World": "oui"}


@app.route("/kpis_1")
def detect_laugh_1():
    filename = "data/humoriste_0_2"
    try:
        prediction_class = Predictor(file=filename)
        prediction_class.get_class()
        scan_class = Scanner(prediction_class.segmented_signal, prediction_class.sampling_freq)
        df_res = scan_class.scan()
        return df_res.to_json()
    except:
        logging.error(traceback.format_exc())


@app.route("/kpis_2", methods=["POST"])
def detect_laugh_2():
    try:
        audio_proprieties = request.json
        prediction_class = Predictor(audio_proprieties=audio_proprieties)
        prediction_class.get_class()
        scan_class = Scanner(prediction_class.segmented_signal, prediction_class.sampling_freq)
        df_res = scan_class.scan()
        return df_res.to_json()
    except:
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    app.run(ssl_context="adhoc", host="0.0.0.0", port=3000)
