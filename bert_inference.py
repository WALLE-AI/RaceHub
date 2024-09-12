import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np



pretrained_model_path= "/root/autodl-tmp/models/chinese-macbert-large"
checkpoint_dir = "/root/autodl-tmp/models/checkpoint_dir/qwen/checkpoint-1342"

def read_or_download_train_test_dataset():
    train_data = pd.read_csv('data/train.csv', sep='\t', header=None)
    test_data = pd.read_csv('data/test.csv', sep='\t', header=None)
    train_data[1],intent_label = pd.factorize(train_data[1])
    return train_data,test_data,intent_label

train_data,test_data,intent_label = read_or_download_train_test_dataset()


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


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_path)
test_encoding = tokenizer(list(test_data[0]), truncation=True, padding=True, max_length=30)
test_dataset = IntentDataset(test_encoding, [0] * len(test_data))
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


label = len(intent_label)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prediction(model, test_dataloader):
    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        pred.append(logits)
        # pred += list(np.argmax(logits, axis=1).flatten())

    return np.vstack(pred)
#'model_0.pt', 'model_1.pt', 'model_2.pt', 'model_3.pt', 
def bert_train_inference():
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_path,num_labels=label)
    model.to(device)
    pred = np.zeros((3000, 12))
    for path in ['model_4.pt']:
        model_path = "/root/autodl-tmp/models/checkpoint_dir/bert"+"/"+path
        ##加载模型
        model.load_state_dict(torch.load(model_path))
        pred += prediction(model, test_dataloader)
    pd.DataFrame({
        'ID': range(1, len(test_data) + 1),
        'Target': [intent_label[x] for x in pred.argmax(1)],
    }).to_csv('nlp_submit.csv', index=None)


def bert_no_train_inference():

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_path)
    model.to(device)
        # model.load_state_dict(torch.load(model_path))
    pred = prediction(model, test_dataloader)
    pd.DataFrame({
    'ID': range(1, len(test_data) + 1),
    'Target': [intent_label[x] for x in pred.argmax(1)],
    }).to_csv('nlp_submit.csv', index=None)
    
if __name__ == "__main__":
    bert_train_inference()