import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/tables/full_comparison.csv')
datasets = df['dataset'].unique()
metrics = ['micro_f1', 'macro_f1', 'hamming_loss']

for dataset in datasets:
    subset = df[df['dataset'] == dataset]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(subset['model'], subset[metric], color=[
                'steelblue', 'coral', 'mediumseagreen', 'gold'])
        plt.title(f'{metric.replace("_", " ").title()} - {dataset.title()}')
        plt.xlabel('Model')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/figures/{metric}_{dataset}.png', dpi=300)
        plt.close()
        print(f"Saved: results/figures/{metric}_{dataset}.png")
