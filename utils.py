import os
import pickle
import torch
from tqdm import tqdm
from importlib import import_module


x = import_module('bert_config')
config = x.Config()

def load_data(config, data):
    CLS, SEP = '[CLS]', '[SEP]'
    contents = []
    with open(data, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            text = eval(line.strip())
            text_a = text['source'].strip()
            text_b = text['target'].strip()
            label = text['labelA'].strip()
            token_a = config.tokenize.tokenize(text_a)
            token_b = config.tokenize.tokenize(text_b)
            tokens = [CLS] + token_a + [SEP]
            token_type_ids = [0] * len(tokens)
            tokens += token_b + [SEP]
            token_type_ids += [1] * (len(token_b) + 1)
            tokens_ids = config.tokenize.convert_tokens_to_ids(tokens)
            seq_len = len(tokens)
            mask = []
            pad_size = config.pad_size
            if seq_len < pad_size:
                mask = [1] * seq_len + [0] * (pad_size-seq_len)
                tokens_ids = tokens_ids + [0] * (pad_size-seq_len)
                token_type_ids = token_type_ids + [0] * (pad_size-seq_len)
            else:
                mask = [1] * pad_size
                tokens_ids = tokens_ids[:pad_size]
                token_type_ids = token_type_ids[:pad_size]
            contents.append((tokens_ids, token_type_ids, mask, int(label)))
    return contents



# load_data(config, config.train_path)


def build_data(config):
    if os.path.exists(config.data_pkl):
        data = pickle.load(open(config.data_pkl, 'rb'))
        train_data = data['train_data']
        dev_data = data['dev_data']
    else:
        data = {}
        train_data = load_data(config, config.train_path)
        dev_data = load_data(config, config.valid_path)
        data['train_data'] = train_data
        data['dev_data'] = dev_data
        pickle.dump(data, open(config.data_pkl, 'wb'))
    return train_data, dev_data

class DatasetIteratior(object):
    def __init__(self, batch, data, device):
        self.data = data
        self.device = device
        self.batch = batch
        self.n_batch = len(data) // batch
        self.index = 0
        self.device = device
        self.residue = False
        if len(data) % self.n_batch != 0:
            self.residue = True

    def to_tensor(self, datas):
        token_ids = torch.LongTensor([item[0] for item in datas]).to(self.device)
        token_type_ids = torch.LongTensor([item[1] for item in datas]).to(self.device)
        mask = torch.LongTensor([item[2] for item in datas]).to(self.device)
        label = torch.LongTensor([item[3] for item in datas]).to(self.device)
        return (token_ids, token_type_ids, mask), label


    def __next__(self):
        if self.residue and self.index == self.n_batch:
            batchs = self.data[self.index*self.batch:len(self.data)]
            self.index += 1
            batchs = self.to_tensor(batchs)
            return batchs
        elif self.index > self.n_batch:
            self.index += 1
            raise StopIteration
        else:
            batchs = self.data[self.index*self.batch:(self.index+1)*self.batch]
            self.index += 1
            batchs = self.to_tensor(batchs)
            return batchs

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batch + 1
        else:
            return self.n_batch

def build_iterator(config, data):
    iter = DatasetIteratior(config.batch_size, data, config.device)
    return iter




























