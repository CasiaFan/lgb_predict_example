import pandas as pd
import numpy as np

input_file = "test_call_history.csv"
res_file = "result.csv"
output_file = "test_call_history_new.csv"

df = pd.read_csv(input_file)
res = pd.read_csv(res_file).set_index("id")
res_dict = res.to_dict()["result"]

df_cp = df.copy().set_index("id")
df_dict = df_cp.T.to_dict('list')

val_idx = []
val_value = []
for key in res_dict.keys():
    if key in df_dict:
        val_idx.append(key)
        val_value.append(res_dict[key])
val_df = df_cp.loc[val_idx]
print(len(val_df))
print(len(val_value))
val_df["result"] = pd.Series(val_value, index=val_df.index)
val_df["id"] = pd.Series(val_idx, index=val_df.index)
val_df.to_csv(output_file, index_label=None, index=None)

