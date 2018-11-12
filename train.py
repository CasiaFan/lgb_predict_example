import xgboost
import lightgbm as lgb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from preprocess import PREPROCESSING_FACTORY as factory
import argparse


def prepare_train_input(input_file, col_name_used):
    """
    Prepare input sample for xgboost classifier training
    """
    if not input_file.endswith("csv"):
        raise ValueError("Input file must be a csv file!")
    if not os.path.exists(input_file):
        raise ValueError("Input file not found! Check input path!")
    df = pd.read_csv(input_file)
    encoded_data = []
    encoders_dict = {}
    cate_dict = {}
    for col_name in col_name_used:
        if col_name != "industry_code":
            (encoding, encoder, categories) = factory[col_name]["fn"](df[col_name], **factory[col_name].get("params", {}))
        else:
            (encoding, encoder, categories) = factory[col_name]["fn"](df[factory[col_name]["cols"]])
        encoded_data.append(encoding)
        encoders_dict[col_name] = encoder
        cate_dict[col_name] = categories
    encoded_data = np.concatenate(encoded_data, axis=1)
    print(encoded_data[:1])
    label = np.asarray(df["result"])
    seed = 7
    test_size = 0.25
    x_train, x_test, y_train, y_test = train_test_split(encoded_data, label, test_size=test_size, random_state=seed)
    return (x_train, x_test, y_train, y_test)


def xgb_train(x_train, y_train, x_test, y_test, model_name):
    xgb_params = dict(max_depth=5,
                      gamma=0.25,
                      learning_rate=0.05,
                      n_estimators=200,
                      booster='dart',
                      objective="binary:logistic",
                      sample_type='uniform',
                      normalize_type="tree",
                      rate_drop=0.1,
                      skip_drop=0.5,
                      colsample_bytree=0.3,
                      subsample=0.7,
                      scale_pos_weight=40,
                      early_stopping_rounds=10,
                      min_child_weight=5,
                      max_delta_depth=0)
    classifier = xgboost.XGBClassifier(**xgb_params)
    classifier.fit(x_train, y_train, eval_metric=['auc'], eval_set=[(x_test, y_test)])
    print("classifier info: ", classifier)
    classifier.save_model(model_name)


def lgb_train(x_train, y_train, x_test, y_test, model_name):
    params = {"num_leaves": 51,
              'num_trees': 150,
              'objective': 'binary',
              'metric': 'auc',
              'max_bin': 20,
              # 'bagging_fraction': 0.8,
              # 'bagging_freq': 5,
              # 'feature_fraction': 0.9,
              'learning_rate': 0.05,
              'num_iterations': 250,
              'max_depth': 5,
              'min_child_weight': 10,
              'colsample_bytree': 0.3,
              'subsample': 0.7,
              'boosting_type': 'dart',
              'scale_pos_weight': 40,
              'boost_from_average': True}
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, valid_sets=lgb_eval)
    print("classifier info: ", gbm)
    gbm.save_model(model_name)


def xgb_eval(x, y, model_file):
    classifier = xgboost.XGBClassifier()
    classifier.load_model(model_file)
    booster = xgboost.Booster()
    booster.load_model(model_file)
    classifier._Booster = booster
    classifier._le = LabelEncoder().fit(y)
    pred = classifier.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, pred)
    print("auc score: ", auc)


def lgb_eval(x, y, model_file):
    classifier = lgb.Booster(model_file=model_file)
    pred = classifier.predict(x, num_iteration=classifier.best_iteration)
    auc = roc_auc_score(y, pred)
    print("auc score: ", auc)


def run(col_name_used, args):
    input_file = args.input_file

    # total potential column names
    # col_name_used = ["week", "charger_id", "call_time", "list_type", "re_call_date", "address",
    #                  "kabushiki_code", "establishment", "shihonkin", "employee_num",
    #                  "kojokazu", "jigyoshokazu", "industry_code", "zenkikessan_uriagedaka",
    #                  "zenkikessan_riekikin", "tokikessan_uriagedaka",
    #                  "tokikessan_riekikin", "tokiuriage_shinchoritsu",
    #                  "zenkiuriage_shinchoritsu", "tokirieki_shinchoritsu",
    #                  "zenkirieki_shinchoritsu", "birthday", "danjokubun",
    #                  "eto_meisho", "tosankeireki", "race_area"]

    model_name = args.model_name
    model_type = args.model_type
    (x_train, x_test, y_train, y_test) = prepare_train_input(input_file, col_name_used=col_name_used)
    if model_type == "xgb":
        xgb_train(x_train, y_train, x_test, y_test, model_name)
        xgb_eval(x_test, y_test, model_name)
    else:
        lgb_train(x_train, y_train, x_test, y_test, model_name)
        lgb_eval(x_test, y_test, model_name)


COL_NAMES = ["week", "call_time", "list_type", "re_call_date", "address",
             "kabushiki_code", "establishment", "kojokazu", "shihonkin", "employee_num",
             "tokikessan_riekikin", "tokikessan_uriagedaka",
             "tokirieki_shinchoritsu", "race_area", "charger_id"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="train_call_history.csv", help="train input file name")
    parser.add_argument("--model_name", default="classifier.model", help="xgboost classifier save name")
    parser.add_argument("--model_type", default="lgb", help="Select classifier: lgb(lightgbm) or xgb(xgboost)", choices=["lgb", "xgb"])
    args = parser.parse_args()
    run(COL_NAMES, args)