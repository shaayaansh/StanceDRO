import pandas as pd 
import numpy as np 
import os

curr_dir = os.getcwd()
data_path = os.path.join(curr_dir, "Data/SemEval-T6/all_train.csv")

df = pd.read_csv(data_path)
atheist_data = df[df["target"] == "Atheism"]
climate_data = df[df["target"] == "Climate Change is a Real Concern"]
feminist_data = df[df["target"] == "Feminist Movement"]

atheist_data.to_csv(os.path.join(curr_dir, "Data/ath_train.csv"))
climate_data.to_csv(os.path.join(curr_dir, "Data/cc_train.csv"))
feminist_data.to_csv(os.path.join(curr_dir, "Data/fm_train.csv"))