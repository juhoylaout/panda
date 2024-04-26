import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Define parameters from the previous script setup
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
checkpoint_iterations = [1547, 3094, 4641]
checkpoint_iterations = [4641]

# Placeholder for evaluation results
evaluation_results = []

# Load CSV files and calculate metrics
for model_name in model_names:
    for checkpoint_iteration in checkpoint_iterations:
        file_name = f"./{project_name}/{model_name}_{checkpoint_iteration}_predictions.csv"
        #file_name = f"./{project_name}/{model_name}_predictions.csv"
        try:
            # Load the results
            results_df = pd.read_csv(file_name)

            # Compute accuracy and classification report
            accuracy = accuracy_score(results_df['Actual_Label'], results_df['Predicted_Label'])
            report = classification_report(results_df['Actual_Label'], results_df['Predicted_Label'], digits=2)
            
            # Append results
            #evaluation_results.append({
            #    'Model': model_name,
            #    'Checkpoint': checkpoint_iteration,
            #    'Accuracy': accuracy,
            #    'Report': report
            #})
            print("#########################")
            print('Model', model_name)
            print(report)

        except Exception as e:
            # In case the file doesn't exist or other error
            evaluation_results.append({
                'Model': model_name,
                'Checkpoint': checkpoint_iteration,
                'Error': str(e)
            })


for res in evaluation_results:
    print(res)
    print("#########################")