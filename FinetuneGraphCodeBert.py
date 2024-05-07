# !pip install transformers==4.30
# !pip install transformers[torch]
# !pip install accelerate -U


# import wandb

# wandb.login(key='edd39beb95ba8985201c7641334b5bbe8c74afc8')


import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from IPython.display import clear_output

print(torch.cuda.is_available())



train =  []
train_label = []

path_to_folder = r"path/to/dataset"
idx = 1
for cwe in os.listdir(path_to_folder):
    folders = os.listdir(os.path.join(path_to_folder, cwe))

    bad_path = os.path.join(path_to_folder,cwe,'bad')
    good_path = os.path.join(path_to_folder,cwe,'good')
    #path to bad and good sources
    bad = os.listdir(bad_path)
    good = os.listdir(good_path)

    for file in bad:
      print(f"Bad {idx}")
      idx+=1
      path = os.path.join(bad_path, file)
      with open(path) as f:
            source = f.read()
      train.append(source)
      train_label.append(1.0 if 'bad' in path else 0.0)
      clear_output()


    for file in good:
      path = os.path.join(good_path, file)
      print(f"good {idx}")
      idx+=1
      with open(path) as f:
            source = f.read()
      train.append(source)
      train_label.append(1.0 if 'good' in path else 0.0)
      clear_output()

X = train 
y = train_label 

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.to(device)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("******************************")
print("*******ENCODING DATASET*******")
print("******************************")
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)


from torch.utils.data import Dataset
from tensorflow.keras.utils import to_categorical

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


y_train_one_hot = to_categorical(y_train, num_classes=2)
y_test_one_hot = to_categorical(y_test, num_classes=2)

train_dataset = CustomDataset(train_encodings, y_train_one_hot)
test_dataset = CustomDataset(test_encodings, y_test_one_hot)

from transformers import  Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

print("******************************")
print("**********FINETUNING**********")
print("******************************")

from transformers import RobertaForSequenceClassification

# with training_args.strategy.scope():
model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

print("******************************")
print("*******SAVING MODEL*******")
print("******************************")
trainer.save_model("/kaggle/working/FTGraphCodeBert.pt")
tokenizer.save_pretrained("/kaggle/working/FTGraphCodeBert.pt")
