import torch
import os
from pytorch_pretrained import BertTokenizer, BertModel
import torch.nn as nn


class Config():
    def __init__(self):
        self.train_path = 'data/train.txt'
        self.valid_path = 'data/valid.txt'
        self.init_bert_path = 'bert_pretrain'
        self.save_model = 'save_model'
        self.require_improvement = False
        self.hidden_size = 768
        self.epoch = 5
        self.lr = 1e-5
        self.tokenize = BertTokenizer.from_pretrained(self.init_bert_path)
        self.num_class = 2
        self.data_pkl = 'data/data_pkl'
        self.pad_size = 128
        self.batch_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, Config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(Config.init_bert_path)
        params_list = self.bert.parameters()
        for param in params_list:
            param.requires_grad = True
        self.fc = nn.Linear(Config.hidden_size, Config.num_class)

    def forward(self, x):
        context = x[0]
        token_type_ids = x[1]
        mask = x[2]
        _, pooled_output = self.bert(context, token_type_ids=token_type_ids, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled_output)
        return out




