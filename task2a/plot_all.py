import pandas as pd
import matplotlib.pyplot as plt
import os

# Find all loss data CSV files in the current directory
loss_files = [f for f in os.listdir('.') if f.startswith('loss_data_rank_') and f.endswith('.csv')]

plt.figure(figsize=(10, 6))

# Plot each rank's loss curve
for file in loss_files:
    # Extract rank number from filename
    rank = file.split('_')[-1].split('.')[0]
    
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Plot the loss curve for this rank
    plt.plot(df['Step'], df['Loss'], label=f'Rank {rank}')

# Add labels and title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Across Different Ranks')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the figure
plt.savefig('loss_curves_all.png')
plt.show()
