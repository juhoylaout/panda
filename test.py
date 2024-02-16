import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm.auto import tqdm
from datasets import Dataset
import pandas as pd

# Parameters
project_name = "toxic-classifier"
model_name = "google/flan-t5-base"
saved_model_name = "./train/toxic-classifier/checkpoint-6164"
batch_size=32
file_path = 'data/test_2024.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(saved_model_name).to(device)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

# Load testing dataset
df = pd.read_csv(file_path, quoting=3)
test_dataset = Dataset.from_pandas(df)
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

def predict(model, dataloader):
    model.eval()
    predictions = []
    progress_bar = tqdm(dataloader, desc="Creating Test Labels")
    with torch.no_grad():
        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            torch.cuda.synchronize()
    return predictions


data_collator = default_data_collator
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

# Run test
with torch.inference_mode():
    predicted_labels = predict(model, test_dataloader)


# Save results
test_dataset = test_dataset.remove_columns(['input_ids', 'attention_mask', 'text', 'label'])
test_dataset = test_dataset.add_column("label", predicted_labels)
test_dataset.to_pandas().to_csv("test_predictions.csv", index=False)

print("Predictions saved to test_predictions.csv")
