import pandas as pd
from pathlib import Path
import os

os.chdir("/Users/au740615/Documents/projects/model_identifier/")

rows = []
for file in Path('./output').glob('metrics_*.csv'):
    df = pd.read_csv(file)
    scenario = file.stem.replace('metrics_', '')
    last = df.iloc[-1]  # last epoch
    for col in df.columns:
        if col.startswith('acc_'):
            model = col.replace('acc_', '')
            acc = last[col]
            rows.append({'scenario': scenario, 'model': model, 'accuracy': acc})

summary = pd.DataFrame(rows)
summary.to_csv('./output/model_accuracy_summary.csv', index=False)
print(summary)
