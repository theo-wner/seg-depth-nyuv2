import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read the training iou and ece logs from the csv files into pandas dataframes
smaller = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp3/seg/smaller_ece.csv')
normal = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp1/results_training_seg_ece/exp1-backbone_b3.csv')
larger = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp3/seg/larger_ece.csv')

# Convert the dataframes to numpy arrays and multiply the iou values by 100 to get percentages
steps = smaller['Step'].to_numpy()
smaller = smaller['Value'].to_numpy()
normal = normal['Value'].to_numpy()
larger = larger['Value'].to_numpy()

# Set the font size and family
plt.rcParams['font.size'] = '26'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Adjust the margins of the figure
fig.subplots_adjust(left=0.16, right=0.85, top=0.94, bottom=0.16)

# Plot the iou for each backbone
ax1.plot(steps, larger, color='#EDB120', label='$1 \cdot 10^{-4}$')  # Darker yellow
ax1.plot(steps, normal, color='#D95319', label='$6 \cdot 10^{-5}$')  # Darker orange
ax1.plot(steps, smaller, color='#0072BD', label='$1 \cdot 10^{-5}$')  # Darker blue

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'.replace('.', ',')))

# Change the x Axis format to k - thousands
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 1e-3:g}k'))

# Add grid lines each 2.5 percent
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.grid(color='gray', linestyle='dashed')

# Set lower and upper limits for the y-axis
ax1.set_ylim(bottom=0.10, top=max(normal) + 0.005)

# Add Labels
ax1.set_xlabel('Iteration')
ax1.set_ylabel('ECE')

# Add a legend to the bottom right
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='lower right', ncol=1)

# Increase dpi for higher resolution
plt.savefig('./images/exp3_seg_ece_training', dpi=300)

# Show the plot
plt.show()

