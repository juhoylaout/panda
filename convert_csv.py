import pandas as pd

# Load the CSV file
df = pd.read_csv('./data/train_2024.csv')

# Rename target labels
df.rename(columns={'target': 'label'}, inplace=True)

# Drop the 'id' column
df.drop('id', axis=1, inplace=True)

# Write the transformed data to a new CSV file
df.to_csv('./data/train.csv', index=False)
