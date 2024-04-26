from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch
import gc
import random
import numpy as np
from accelerate.utils import release_memory
import schedulefree
import pandas as pd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

device = "cuda" if torch.cuda.is_available() else "cpu"
project_name = "toxic-classifier"
model_names = [
    "bert-base-uncased",
    "bert-large-uncased",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "google/flan-t5-base"
]

max_length = 256
batch_size = 16
gradient_accumulation = 4
learning_rate = 4e-5
num_train_epochs = 3
num_warmup_steps = 100
data_files = {'train': '../data/train_2024.csv', 'validation': '../data/dev_2024.csv'}
id2label = {0: "normal", 1: "toxic"}
label2id = {"normal": 0, "toxic": 1}

df = pd.read_csv(data_files['train'], quoting=3).rename(columns={'target': 'label'})
train_dataset = Dataset.from_pandas(df).shuffle(seed=SEED)

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id).to(device)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate, warmup_steps=num_warmup_steps)
    
    training_args = TrainingArguments(
        output_dir=f"{project_name}/{model_name}",
        group_by_length=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=num_train_epochs,
        logging_dir=f"{project_name}/{model_name}/logs",
        logging_strategy="steps",
        logging_steps=200,
        save_strategy="epoch",
        bf16=True,
        tf32=True,
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        args=training_args,
        train_dataset=train_dataset,
    )
    
    if device == "cuda":
        flush()
        release_memory(model)
    
    trainer.train()

    if device == "cuda":
        del model
        del tokenizer
        del trainer
        del optimizer
        flush()

    print(f"Finished training {model_name}")
