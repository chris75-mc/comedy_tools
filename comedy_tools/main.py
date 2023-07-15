import argparse
from src.predictor import Predictor
from src.scanner import Scanner

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="Location of the audio file")

args = parser.parse_args()


def main():
    print("Hey : ", args)
    if args.filename:
        prediction_class = Predictor(args.filename)
        prediction_class.get_class()
        scan_class = Scanner(prediction_class.segmented_signal, prediction_class.sampling_freq)
        df_res = scan_class.scan()
        print(df_res)
        return "ok"
    return "No file name"


if __name__ == "__main__":
    main()
