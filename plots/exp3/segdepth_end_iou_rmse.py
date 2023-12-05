import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set the font size and family
plt.rcParams['font.size'] = '26'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
lrs = ['$1 \cdot 10^{-5}$', '$5 \cdot 10^{-5}$', '$1 \cdot 10^{-4}$']
iou = np.array([0.5028, 0.5166, 0.5017]) * 100
rmse = np.array([0.5770, 0.5457, 0.5492])

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.14, right=0.86, top=0.97, bottom=0.16)

# Draw points and lines for the first y-axis
ax1.plot(lrs, iou, marker='o', color='#0072BD', label='mIoU')  # Darker blue

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(lrs, rmse, marker='o', color='#D95319', label='RMSE')  # Darker yellow

# Für die linke Y-Achse (ax1)
ax1.tick_params(axis='y', colors='#0072BD')  # Dunkler Blau
ax1.set_ylabel('mIoU in \%', color='#0072BD')

# Für die rechte Y-Achse (ax2)
ax2.tick_params(axis='y', colors='#D95319')  # Dunkler Gelb
ax2.set_ylabel('RMSE in m', color='#D95319')

# Get current y-axis limits
y1_lim = ax1.get_ylim()
y2_lim = ax2.get_ylim()

# Determine the number of ticks you want on the axes
num_ticks = 5

# Use numpy linspace to generate evenly spaced ticks
ticks1 = np.linspace(y1_lim[0], y1_lim[1], num_ticks)[1:]  # Skip the first value
ticks2 = np.linspace(y2_lim[0], y2_lim[1], num_ticks)[1:]  # Skip the first value

# Set the ticks on the axes
ax1.set_yticks(ticks1)
ax2.set_yticks(ticks2)

# Set the limits of the axes
ax1.set_ylim(y1_lim)
ax2.set_ylim(y2_lim)

# Add Grid
ax1.grid(True, color='grey', linestyle='--')

# Add labels to the axes
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('mIoU in \%')
ax2.set_ylabel('RMSE in m')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'))

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Increase dpi for higher resolution
plt.savefig('./images/exp3_segdepth_end_iou_rmse.png', dpi=300)

# Show the plot
plt.show()