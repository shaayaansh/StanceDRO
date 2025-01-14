from model import StanceDetector
from dataset import StanceDataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.optim import AdamW

curr_path = os.getcwd()
data_path = os.path.join(curr_path, "Data")

model_name = "bert-base-uncased"

# loading the dataframes
ath_data = pd.read_csv(os.path.join(data_path, "ath_train.csv"))    # 513 rows
cc_data = pd.read_csv(os.path.join(data_path, "cc_train.csv"))      # 395 rows
fm_data = pd.read_csv(os.path.join(data_path, "fm_train.csv"))      # 664 rows

# lets take only 350 from each dataset
# to make a fully balanced dataset
ath_data = ath_data[:350]
cc_data = cc_data[:350]
fm_data = fm_data[:80]

# convert dataframes to dataset objects
ath_dataset = StanceDataset(ath_data, model_name)
cc_dataset = StanceDataset(cc_data, model_name)
fm_dataset = StanceDataset(fm_data, model_name)

ath_dataloader = DataLoader(ath_dataset, batch_size=4, shuffle=True)
cc_dataloader = DataLoader(cc_dataset, batch_size=4, shuffle=True)
fm_dataloader = DataLoader(fm_dataset, batch_size=4, shuffle=True)

# initialize q 
q = torch.tensor([0.333, 0.333, 0.333])
lr_q = 0.01
lr_m = 1e-5
model = StanceDetector(model_name)
num_iterations = 300
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr_m)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model.to(device)
q.to(device)
model.train()

# train loop
for iteration in range(num_iterations):
    g = np.random.randint(0, 3) # generate a random number {0, 1, 2}
    if g == 0:
        tokenized, labels = next(iter(ath_dataloader))
    elif g == 1:
        tokenized, labels = next(iter(cc_dataloader))
    else:
        tokenized, labels = next(iter(fm_dataloader))


    input_ids = tokenized["input_ids"].squeeze(1).to(device)
    attention_mask = tokenized["attention_mask"].squeeze(1).to(device)
    labels = labels.squeeze(-1).to(device)
    y_pred = model(input_ids, attention_mask)
    loss = loss_fn(y_pred, labels)

    q[g] *= torch.exp(lr_q*loss)    # update weights 
    q = q / q.sum()                 # normalize the weights

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_m * q[g]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if g == 2:
        print("LOSS: ", loss.item())
        print("Group Weight: ", q[g])

    #print(q)


    