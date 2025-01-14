import pandas as pd
import numpy as np
from transformers import AutoModel
import torch.nn as nn

class StanceDetector(nn.Module):
    def __init__(self, model_name, dropout=0.4):
        super(StanceDetector, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(hidden_size, 3) # there are 3 labels in the task: pro, con, neutral

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask).last_hidden_state[:,0,:]
        out = self.dropout(out)
        out = self.linear(out)

        return out