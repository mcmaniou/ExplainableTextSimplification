# !pip install transformers
# !pip install transformers[torch]
# !pip install accelerate -U

# Importing the libraries needed
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch import tensor
from torch.utils.data import Dataset
import accelerate

import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

# Define key variables that will be used later
train_batch_size = 8
valid_batch_size = 8
epochs = 20
data_filepath = "sentences_data.csv"
output_dir = "bert/model_sentences"
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', max_length=512)

df=pd.read_csv(data_filepath)

# Compute metrics in test subset function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds, digits=3, target_names=['Class 0','Class 1']))
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Train - test - validation data split
X_all = df['TEXT'].to_numpy()
y_all = df['CATEGORY'].to_numpy()

X_train, X_test_val, y_train, y_test_val = train_test_split(X_all, y_all, test_size=0.4, random_state=42, stratify=y_all)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25, random_state=42, stratify=y_test_val)

print("Train set: " +  str(len(X_train)))
print("Test set: " +  str(len(X_test)))
print("Validation set: " +  str(len(X_val)))

# Prepare data for model
class myDataset(Dataset):
  def __init__(self, encodings, labels, tokenizer):
    self.encodings = tokenizer(list(encodings), truncation=True, padding=True)
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

train_dataset = myDataset(X_train, y_train, tokenizer)
validation_dataset = myDataset(X_val, y_val, tokenizer)

# Finetune pubmed bert
model = BertForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=2)

training_arguments = TrainingArguments(evaluation_strategy='epoch',
                                       save_strategy='epoch',
                                       output_dir="./results",
                                       num_train_epochs=epochs,
                                       save_total_limit = 2,
                                       per_device_train_batch_size=train_batch_size,
                                       per_device_eval_batch_size=valid_batch_size,
                                       warmup_steps=200,
                                       weight_decay=0.01,
                                       load_best_model_at_end=True)

trainer = Trainer(model=model, 
                  args=training_arguments, 
                  train_dataset=train_dataset,
                  eval_dataset=validation_dataset, 
                  compute_metrics=compute_metrics)

trainer.train()

# Save model
trainer.model.save_pretrained(output_dir)

# Results in test subset
test_dataset = myDataset(X_test, y_test, tokenizer)
trainer.predict(test_dataset)