from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments,
    Trainer, 
    DataCollatorWithPadding
)
from datasets import load_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
project_name="toxic-classifier"
model_name = "google/flan-t5-base"
batch_size = 8
gradient_accumulation = 4
mixed_precision = "bf16"
num_train_epochs = 2
num_warmup_steps = 400
data_files = {'train': '../data/train.csv', 'validation': '../data/dev.csv'}
dataset = load_dataset('csv', data_files=data_files)
id2label = {0: "NORMAL", 1: "TOXIC"}
label2id = {"NORMAL": 0, "TOXIC": 1}

# load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512)

dataset_size = len(tokenized_datasets["train"])
effective_batch_size = batch_size * gradient_accumulation
steps_per_epoch = dataset_size / effective_batch_size
eval_steps = steps_per_epoch / 10

training_args = TrainingArguments(
    output_dir=project_name,
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    num_train_epochs=num_train_epochs,
    warmup_steps=num_warmup_steps,
    weight_decay=0.01,
    logging_dir=f'{project_name}/logs',
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    bf16=True,
    tf32=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

# Train the model
trainer.train()