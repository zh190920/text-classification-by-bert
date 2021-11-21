import torch
import os
import numpy as np
from tqdm import tqdm
from pytorch_pretrained import BertAdam, optimization
import torch.nn.functional as F
import utils
from sklearn import metrics

def train(config, model, train_iter, dev_iter):
    params_list = model.named_parameters()
    no_decay = ['bias', 'LayerNorm.weight', 'layerNorm.bias']
    optimizer_group_parameters = [
        {'params': [p for n,p in params_list if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params':[p for n,p in params_list if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]
    print(len(train_iter))
    optimizer = BertAdam(optimizer_group_parameters, lr=config.lr, warmup=0.05, t_toatal=len(train_iter)*config.epoch)
    model.train()
    step = 0
    dev_best_loss = 0
    last_improvement = 0
    improvement = ""
    flag = False
    for epoch in range(config.epoch):
        print("epoch:{}/{}".format(epoch, config.epoch))
        for i,(train_data, label) in tqdm(enumerate(train_iter)):
            out = model(train_data)
            model.zero_grad()
            loss =F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()
            if step % 100 ==0:
                label = label.data.cpu()
                pred = torch.max(out.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(label, pred)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    model_name = 'save_{}/{}'.format(epoch, step) + '.ckpt'
                    save_model_path = os.path.join(config.save_model, model_name)
                    torch.save(model, save_model_path)
                    last_improvement += 1
                    improvement = "up"
                else:
                    improvement = 'no up'
                msg = "Iter:{}/{}, train_acc:{}, train_loss:{},  dev_acc:{}, dev_loss:{}"
                print(msg.format(epoch, step, train_acc, loss.item(), dev_acc, dev_loss))
            step += 1
            if step - last_improvement > config.require_improvement:
                flag = True
                break
        if flag:
            break
def evaluate(config, model, dev_iter):
    model.eval()
    pre_all = np.array([], dtype=int)
    label_all = np.array([], dtype=int)
    loss_tatal = 0.0
    with torch.no_grad():
        for dev_data, label in dev_iter:
            out = model(dev_data)
            loss = F.cross_entropy(out, label)
            label = label.data.cpu().numpy()
            loss_tatal += loss
            predict = torch.max(out.data, 1)[1].cpu().numpy()
            pre_all = np.append(pre_all, predict)
            label_all = np.append(label_all, label)

        acc = metrics.accuracy_score(label_all, pre_all)
        loss = loss_tatal / len(dev_iter)
        return acc, loss


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_model))
    model.eval()
    predict_all = []
    label_all = []
    with torch.no_grad():
        for test_data, label in test_iter:
            out = model(test_data)
            predict = torch.max(out, 1)[1].cpu().numpy()
            label = label.data.cpu().numpy()
            predict_all = np.append(predict_all, predict)
            label_all = np.append(label_all, label)
        acc = metrics.accuracy_score(label_all, predict_all)
        report = metrics.classification_report(label_all, predict_all)
        return acc, report





