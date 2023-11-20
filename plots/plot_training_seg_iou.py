import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define a function to change decimal points to commas
def comma_decimal(x, pos):
    return str(x).replace('.', ',')

# Read the training iou and ece logs from the csv files into pandas dataframes
df_b0 = pd.read_csv('../results/results_training_seg_iou/exp1-backbone_b0.csv')
df_b1 = pd.read_csv('../results/results_training_seg_iou/exp1-backbone_b1.csv')
df_b2 = pd.read_csv('../results/results_training_seg_iou/exp1-backbone_b2.csv')
df_b3 = pd.read_csv('../results/results_training_seg_iou/exp1-backbone_b3.csv')
df_b4 = pd.read_csv('../results/results_training_seg_iou/exp1-backbone_b4.csv')
df_b5 = pd.read_csv('../results/results_training_seg_iou/exp1-backbone_b5.csv')

# Convert the dataframes to numpy arrays and multiply the iou values by 100 to get percentages
steps = df_b0['Step'].to_numpy()
b0_iou = df_b0['Value'].to_numpy() * 100
b1_iou = df_b1['Value'].to_numpy() * 100
b2_iou = df_b2['Value'].to_numpy() * 100
b3_iou = df_b3['Value'].to_numpy() * 100
b4_iou = df_b4['Value'].to_numpy() * 100
b5_iou = df_b5['Value'].to_numpy() * 100

# Set the font size and family
plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'serif'

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Adjust the margins of the figure
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Plot the iou for each backbone
ax1.plot(steps, b0_iou, color='#0072BD', label='b0')  # Darker blue
ax1.plot(steps, b1_iou, color='#D95319', label='b1')  # Darker orange
ax1.plot(steps, b2_iou, color='#EDB120', label='b2')  # Darker yellow
ax1.plot(steps, b3_iou, color='#7E2F8E', label='b3')  # Darker purple
ax1.plot(steps, b4_iou, color='#77AC30', label='b4')  # Darker green
ax1.plot(steps, b5_iou, color='#4DBEEE', label='b5')  # Darker cyan

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(comma_decimal))

# Add grid lines each 2.5 percent
ax1.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax1.grid(color='gray', linestyle='dashed')

# Set lower and upper limits for the y-axis
ax1.set_ylim(bottom=30, top=max(b5_iou) + 0.5)

# Add Labels
ax1.set_xlabel('Iteration')
ax1.set_ylabel('mIoU in %')

# Add a legend to the bottom right
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='lower right', ncol=2)

# Increase dpi for higher resolution
plt.savefig('training_seg_iou.png', dpi=300)

# Show the plot
plt.show()

