import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read the training iou and ece logs from the csv files into pandas dataframes
df_fff = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp2/segdepth/fff_ece.csv')
df_ttf = pd.read_csv('/home/tkapler/Dokumente/Studium/Bachelorarbeit/Code/seg-depth-nyuv2/results/exp2/segdepth/ttf_ece.csv')

# Convert the dataframes to numpy arrays
steps = df_fff['Step'].to_numpy()
fff = df_fff['Value'].to_numpy()
ttf = df_ttf['Value'].to_numpy()

# Convert the values to percent
fff
ttf

# Set the font size and family
plt.rcParams['font.size'] = '26'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5))

# Adjust the margins of the figure
fig.subplots_adjust(left=0.14, right=0.86, top=0.97, bottom=0.15)

# Plot the iou for each backbone
ax1.plot(steps, fff, color='#0072BD', label='Keine Augmentations')  # Darker blue
ax1.plot(steps, ttf, color='#D95319', label='Standard-Augmentations')  # Darker orange

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'.replace('.', ',')))

# Change the x Axis format to k - thousands
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 1e-3:g}k'))

# Add grid lines each 2.5 percent
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.grid(color='gray', linestyle='dashed')

# Set lower and upper limits for the y-axis
ax1.set_ylim(bottom=0.17, top=max(fff) + 0.002)

# Add Labels
ax1.set_xlabel('Iteration')
ax1.set_ylabel('ECE')

# Add a legend to the bottom right
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='lower right', ncol=1)

# Increase dpi for higher resolution
plt.savefig('./images/segdepth_training_ece_exp2.png', dpi=300)

# Show the plot
plt.show()

