import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define a function to change decimal points to commas
def comma_decimal(x, pos):
    return str(x).replace('.', ',')

# Read the training iou and ece logs from the csv files into pandas dataframes
df_b0 = pd.read_csv('../results/results_training_depth_rmse/exp1-backbone_b0.csv')
df_b1 = pd.read_csv('../results/results_training_depth_rmse/exp1-backbone_b1.csv')
df_b2 = pd.read_csv('../results/results_training_depth_rmse/exp1-backbone_b2.csv')
df_b3 = pd.read_csv('../results/results_training_depth_rmse/exp1-backbone_b3.csv')
df_b4 = pd.read_csv('../results/results_training_depth_rmse/exp1-backbone_b4.csv')
df_b5 = pd.read_csv('../results/results_training_depth_rmse/exp1-backbone_b5.csv')

# Convert the dataframes to numpy arrays
steps = df_b0['Step'].to_numpy()
b0_rmse = df_b0['Value'].to_numpy()
b1_rmse = df_b1['Value'].to_numpy()
b2_rmse = df_b2['Value'].to_numpy()
b3_rmse = df_b3['Value'].to_numpy()
b4_rmse = df_b4['Value'].to_numpy()
b5_rmse = df_b5['Value'].to_numpy()

# Set the font size and family
plt.rcParams['font.size'] = '18'
plt.rcParams['font.family'] = 'serif'

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Adjust the margins of the figure
fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.13)

# Plot the iou for each backbone
ax1.plot(steps, b0_rmse, color='#0072BD', label='b0')  # Darker blue
ax1.plot(steps, b1_rmse, color='#D95319', label='b1')  # Darker orange
ax1.plot(steps, b2_rmse, color='#EDB120', label='b2')  # Darker yellow
ax1.plot(steps, b3_rmse, color='#7E2F8E', label='b3')  # Darker purple
ax1.plot(steps, b4_rmse, color='#77AC30', label='b4')  # Darker green
ax1.plot(steps, b5_rmse, color='#4DBEEE', label='b5')  # Darker cyan

# Change the decimal separator to comma
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(comma_decimal))

# Change the x Axis format to k - thousands
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 1e-3:g}k'))

# Add grid lines each 2.5 percent
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.grid(color='gray', linestyle='dashed')

# Set lower and upper limits for the y-axis
ax1.set_ylim(bottom=min(b5_rmse) - 0.005, top=1)

# Add Labels
ax1.set_xlabel('Iteration')
ax1.set_ylabel('RMSE in m')

# Add a legend to the bottom right
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='upper right', ncol=2)

# Increase dpi for higher resolution
plt.savefig('depth_training_rmse.png', dpi=300)

# Show the plot
plt.show()

