import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np


def eval():
    y_file = "result.csv"
    # y_file = "train_call_history.csv"
    pred_file = "test_pred.txt"

    df1 = pd.read_csv(y_file)[["id", "result"]]
    df1 = df1.set_index("id")
    df2 = pd.read_csv(pred_file).set_index("id")

    df1_dict = df1.to_dict()["result"]
    df2_dict = df2.to_dict()["result"]
    val_res1, val_res2 = [], []
    for key in df2_dict.keys():
        if key in df1_dict:
            val_res1.append(df1_dict[key])
            val_res2.append(df2_dict[key])

    val_res1 = np.array(val_res1)
    val_res2 = np.array(val_res2)
    score = roc_auc_score(val_res1, val_res2)
    print("ros score: ", score)
    return score

eval()