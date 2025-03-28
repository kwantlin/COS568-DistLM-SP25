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

# Calculate average loss across all ranks
all_data = []
for file in loss_files:
    df = pd.read_csv(file)
    all_data.append(df)

# Create a common step index
all_steps = sorted(list(set().union(*[set(df['Step']) for df in all_data])))
avg_loss = []

# Calculate average loss at each step
for step in all_steps:
    losses_at_step = [df.loc[df['Step'] == step, 'Loss'].values[0] 
                     for df in all_data if step in df['Step'].values]
    if losses_at_step:
        avg_loss.append(sum(losses_at_step) / len(losses_at_step))
    else:
        avg_loss.append(None)  # Handle missing steps

# Plot the average loss curve
plt.plot(all_steps, avg_loss, 'k--', linewidth=2, label='Average Loss')

# Add labels and title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Task 2a: Training Loss Across Different Ranks')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the figure
plt.savefig('loss_curves_all.png')
plt.show()

