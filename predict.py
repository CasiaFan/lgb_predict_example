import xgboost
import lightgbm as lgb
import pandas as pd
import numpy as np
from preprocess import PREPROCESSING_FACTORY as factory
from sklearn.preprocessing import LabelEncoder
import argparse, pickle
from train import COL_NAMES


def predict(data, model_name, model_type="lgb"):
    """
    Predict with input data after preprocessing
    """
    col_names = COL_NAMES
    encoded_data = []
    encoders = pickle.load(open("encoder.bin", "rb"))
    for col_name in col_names:
        if not col_name == "industry_code":
            (encoding, encoder, categories) = factory[col_name]["fn"](data[col_name], encoder=encoders[col_name], **factory[col_name].get("params", {}))
        else:
            (encoding, encoder, categories) = factory[col_name]["fn"](data[factory[col_name]["cols"]])
        encoded_data.append(encoding)
    encoded_data = np.concatenate(encoded_data, axis=1)
    if model_type == "xgb":
        classifier = xgboost.XGBClassifier()
        booster = xgboost.Booster()
        booster.load_model(model_name)
        classifier._Booster = booster
        classifier._le = LabelEncoder().fit([0, 1])
        result = classifier.predict_proba(encoded_data)
        result = result[:, 1]
    else:
        classifier = lgb.Booster(model_file=model_name)
        result = classifier.predict(encoded_data)
    return (result, data["id"])


def main(args):
    input_file = args.input_file
    df = pd.read_csv(input_file)
    (res, id) = predict(df, model_name=args.model_name, model_type=args.model_type)
    output_file = args.output_file
    with open(output_file, "w") as f:
        f.write("id,result\n")
        for idx in range(len(id)):
            f.write("{},{}\n".format(id[idx], res[idx]))


def train_main():
    input_file = "test_call_history.csv"
    df = pd.read_csv(input_file)
    (res, id) = predict(df, model_name="classifier.model", model_type="lgb")
    output_file = "test_pred.txt"
    with open(output_file, "w") as f:
        f.write("id,result\n")
        for idx in range(len(id)):
            f.write("{},{}\n".format(id[idx], res[idx]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="test_call_history.csv", help="train input file name")
    parser.add_argument("--model_name", default="classifier.model", help="classifier name")
    parser.add_argument("--model_type", default="lgb", help="Select classifier: lgb(lightgbm) or xgb(xgboost)", choices=["lgb", "xgb"])
    parser.add_argument("--output_file", default="test_pred.txt", help="prediction result save file")
    args = parser.parse_args()
    main(args)


