from pathlib import Path
path = Path("training_results/finetuning/LLM_MOFID/gpt-4o-mini-2024-07-18_MOFID_CoRE2019_1_Di/test_results_Di.csv")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv(path)

# Extract targets and predictions
targets = df['target']
predictions = df['pred']

# Create scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(targets, predictions, alpha=0.6, s=30)

# Set axis limits to start at 0
max_val = max(max(targets), max(predictions))
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# Add perfect prediction line (y=x) from 0 to max
plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')

# Calculate SRCC for display on plot
srcc = spearmanr(targets, predictions)[0]

# Add text annotation with SRCC
plt.text(0.05 * max_val, 0.9 * max_val, f'SRCC: {srcc:.3f}', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Labels and title
plt.xlabel('Targets')
plt.ylabel('Predicted')
plt.title('GPT-4o-mini Prediction on Di')
plt.legend()
plt.grid(True, alpha=0.3)

# Make plot square
plt.axis('equal')
plt.tight_layout()

# Show the plot
plt.show()

# Print some basic statistics
mae = np.mean(np.abs(targets - predictions))
rmse = np.sqrt(np.mean((targets - predictions)**2))
r2 = np.corrcoef(targets, predictions)[0, 1]**2

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Square Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Number of samples: {len(targets)}")

print(f"Spearman Rank Correlation Coefficient (SRCC): {srcc:.3f}")