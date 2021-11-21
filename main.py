import torch
import utils
import argparse
import random
import  numpy as np
import train
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert_config', help='the module name')
args = parser.parse_args()

if __name__=='__main__':
    model_name = args.model
    x = import_module(model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    train_data, dev_data = utils.build_data(config)
    train_iter = utils.build_iterator(config, train_data)
    dev_iter = utils.build_iterator(config, dev_data)
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter)



