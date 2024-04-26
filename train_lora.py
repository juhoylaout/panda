from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments,
    Trainer, 
    DataCollatorWithPadding
)
from datasets import load_dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"
project_name="toxic-classifier"
model_name = "google/flan-t5-large"
batch_size = 32
num_train_epochs = 2
num_warmup_steps = 400
data_files = {'train': '../data/train.csv', 'validation': '../data/dev.csv'}
dataset = load_dataset('csv', data_files=data_files)
id2label = {0: "NORMAL", 1: "TOXIC"}
label2id = {"NORMAL": 0, "TOXIC": 1}

torch.cuda.empty_cache()

# load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id, load_in_8bit=True, device_map="auto")

# Define LoRA Config
peft_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512)



training_args = TrainingArguments(
    output_dir=project_name,
    learning_rate=5e-4,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_steps=num_warmup_steps,
    weight_decay=0.01,
    fp16=True,
    logging_dir=f'{project_name}/logs',
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)
model.config.use_cache = False

# Train the model
trainer.train()