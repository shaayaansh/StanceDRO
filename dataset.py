from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

class StanceDataset(Dataset):
    def __init__(self, dataframe, model_name):
        self.df = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.df.iloc[idx]["text"],
            self.df.iloc[idx]["target"],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        if self.df.iloc[idx]["label"] == "FAVOR":
            label = torch.tensor([0])
        elif self.df.iloc[idx]["label"] == "AGAINST":
            label = torch.tensor([1])
        else:
            label = torch.tensor([2]) 
        
        return tokenized, label
