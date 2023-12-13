import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Read the training iou and ece logs from the csv files into pandas dataframes
df_b0 = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b0.csv')
df_b1 = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b1.csv')
df_b2 = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b2.csv')
df_b3 = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b3.csv')
df_b4 = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b4.csv')
df_b5 = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b5.csv')

# Convert the dataframes to numpy arrays and multiply the iou values by 100 to get percentages
steps = df_b0['Step'].to_numpy()
b0 = df_b0['Value'].to_numpy()
b1 = df_b1['Value'].to_numpy()
b2 = df_b2['Value'].to_numpy()
b3 = df_b3['Value'].to_numpy()
b4 = df_b4['Value'].to_numpy()
b5 = df_b5['Value'].to_numpy()

# Set the font size and family
plt.rcParams['font.size'] = '22'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Adjust the margins of the figure
fig.subplots_adjust(left=0.14, right=0.86, top=0.97, bottom=0.15)

# Plot the iou for each backbone
ax1.plot(steps, b0, color='#0072BD', label='B0')  # Darker blue
ax1.plot(steps, b1, color='#D95319', label='B1')  # Darker orange
ax1.plot(steps, b2, color='#EDB120', label='B2')  # Darker yellow
ax1.plot(steps, b3, color='#7E2F8E', label='B3')  # Darker purple
ax1.plot(steps, b4, color='#77AC30', label='B4')  # Darker green
ax1.plot(steps, b5, color='#4DBEEE', label='B5')  # Darker cyan

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'.replace('.', ',')))

# Change the x Axis format to k - thousands
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 1e-3:g}k'))

# Add grid lines
ax1.grid(color='gray', linestyle='dashed')

# Set lower and upper limits for the y-axis
ax1.set_ylim(bottom=0.1, top=max(b1) + 0.005)

# Add Labels
ax1.set_xlabel('Iteration')
ax1.set_ylabel('ECE')

# Add a legend to the bottom right
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='lower right', ncol=2)

# Save the plot
plt.savefig('./images/seg_training_ece.png', dpi=300)

# Show the plot
plt.show()

