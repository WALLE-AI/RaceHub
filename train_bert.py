import loguru
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
import re

pretrained_model_path= "/root/autodl-tmp/models/chinese-macbert-large"
checkpoint_dir = "/root/autodl-tmp/models/checkpoint_dir/bert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_or_download_train_test_dataset():
    train_data = pd.read_csv('data/train.csv', sep='\t', header=None)
    test_data = pd.read_csv('data/test.csv', sep='\t', header=None)
    loguru.logger.info(f"train data {train_data.head()}")
    ##转标签函数
    train_data[1], intent_label = pd.factorize(train_data[1])
    return train_data,test_data,len(intent_label)

def read_or_download_train_test_dataset_train_test_split():
    train_data = pd.read_csv('data/train.csv', sep='\t', header=None)
    test_data = pd.read_csv('data/test.csv', sep='\t', header=None)
    ##进行数据集与测试切分
    x_train, x_test, train_label, test_label = train_test_split(train_data[0].values,
                                                            train_data[1].values,
                                                            test_size=0.2,
                                                            stratify=train_data[1].values)
    return x_train, x_test, train_label, test_label


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_path)
    # model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_path, num_labels=intent_label)
    # model.to(device)
    return tokenizer

def load_model(intent_label):
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_path, num_labels=intent_label)
    model.to(device)
    return model
    

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


    
    
def build_train_intent_datasets(tokenizer):
    x_train, x_test, train_label, test_label = read_or_download_train_test_dataset_train_test_split()
    train_encoding = tokenizer(list(x_train), truncation=True, padding=True, max_length=30)
    test_encoding = tokenizer(list(x_test), truncation=True, padding=True, max_length=30)
    train_dataset = IntentDataset(train_encoding, train_label)
    test_dataset = IntentDataset(test_encoding, test_label)
    ##显示一下格式
    loguru.logger.info("train data :{train_dataset[1]},train label:{train_lable[1]}")
    return train_dataset,test_dataset

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def train_batch_epoch(model, train_loader, epoch,optim):
    
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optim.step()
        # scheduler.step()

        iter_num += 1
        if(iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss/len(train_loader)))
    
    
def validation_batch_epoch(model, val_dataloader):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in val_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss/len(val_dataloader)))
    print("-------------------------------")


def build_train_intent_dataset_kflod(tokenizer):
    from sklearn.model_selection import KFold
    train_data,test_data,total_label = read_or_download_train_test_dataset()
    kf = KFold(n_splits=5)
    for train_idx, val_idx in kf.split(train_data[0].values, train_data[1].values,):
        print(train_idx)
        train_text = train_data[0].iloc[train_idx]
        val_text = train_data[0].iloc[train_idx]

        train_label = train_data[1].iloc[train_idx].values
        val_label = train_data[1].iloc[train_idx].values

        train_encoding = tokenizer(list(train_text), truncation=True, padding=True, max_length=30)
        val_encoding = tokenizer(list(val_text), truncation=True, padding=True, max_length=30)

        # 默认是没有数据扩增，文本默认是没有变换的操作
        train_dataset = IntentDataset(train_encoding, train_label)
        val_dataset = IntentDataset(val_encoding, val_label)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        yield train_loader,val_dataloader,total_label



def train():
    tokenizer = load_tokenizer()
    fold = 0
    for train_loader,val_dataloader,total_label in build_train_intent_dataset_kflod(tokenizer=tokenizer):
        # 加载每折的模型
        model= load_model(total_label)
        optim = AdamW(model.parameters(), lr=1e-5)
        total_steps = len(train_loader) * 1
        scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

        for epoch in range(1):
            train_batch_epoch(model, train_loader, epoch,optim)
            validation_batch_epoch(model, val_dataloader)

        torch.save(model.state_dict(), checkpoint_dir +'/model_' + str(fold) + '.pt')
        fold += 1
        
if __name__ =="__main__":
    train()






