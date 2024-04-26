import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import gc
import random
import numpy as np

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

project_name = "toxic-classifier"
model_names = [
    "bert-base-uncased",
    "bert-large-uncased",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "siebert/sentiment-roberta-large-english",
    "google/flan-t5-base"
]

batch_size=32
checkpoint_iterations=[1547,3094,4641]
max_input_length = 512
file_path = '../data/train_2024.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(file_path, quoting=3).rename(columns={'target': 'label'})
validation_dataset = Dataset.from_pandas(df)

def evaluate(model, tokenizer, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return actual_labels, predictions

for checkpoint_iteration in checkpoint_iterations:
    print(f"####### checkpoint_iteration: {checkpoint_iteration}   #####")
    for model_name in model_names:
        saved_model_name = f"./{project_name}/{model_name}/checkpoint-{checkpoint_iteration}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length=512)
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_name).to(device)

        def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=max_input_length)

        # Load testing dataset
        validation_dataset = validation_dataset.map(tokenize_function, batched=True)
        data_collator = default_data_collator
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=data_collator)

        # Do evaluation
        actual_labels, predictions = evaluate(model, tokenizer, validation_dataloader, device)

        accuracy = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

        results_df = pd.DataFrame({
            'Actual_Label': actual_labels,
            'Predicted_Label': predictions
        })

        results_df.to_csv(f"./{project_name}/{model_name}_{checkpoint_iteration}_predictions.csv", index=False)

        if device == "cuda":
            del model
            del tokenizer
            flush()

        print(f"Results for {model_name} saved.")
