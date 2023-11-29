import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set the font size and family
plt.rcParams['font.size'] = '22'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']
val_iou = [39.64, 42.33, 47.90, 51.07, 51.64, 51.93]
val_calibration_error = [22.92, 23.87, 22.06, 21.17, 20.98, 20.95]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.12, right=0.88, top=0.97, bottom=0.15)

# Draw points and lines for the first y-axis
ax1.plot(backbones, val_iou, marker='o', color='#0072BD', label='mIoU')  # Darker blue

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(backbones, val_calibration_error, marker='o', color='#D95319', label='ECE')  # Darker yellow

# Für die linke Y-Achse (ax1)
ax1.tick_params(axis='y', colors='#0072BD')  # Dunkler Blau
ax1.set_ylabel('Trainingszeit in h', color='#0072BD')

# Für die rechte Y-Achse (ax2)
ax2.tick_params(axis='y', colors='#D95319')  # Dunkler Gelb
ax2.set_ylabel('Inferenzzeit in s', color='#D95319')

# Get current y-axis limits
y1_lim = ax1.get_ylim()
y2_lim = ax2.get_ylim()

# Determine the number of ticks you want on the axes
num_ticks = 5

# Use numpy linspace to generate evenly spaced ticks
ticks1 = np.linspace(y1_lim[0], y1_lim[1], num_ticks)
ticks2 = np.linspace(y2_lim[0], y2_lim[1], num_ticks)

# Set the ticks on the axes
ax1.set_yticks(ticks1)
ax2.set_yticks(ticks2)

# Set the limits of the axes
ax1.set_ylim(y1_lim)
ax2.set_ylim(y2_lim)

# Add Grid
ax1.grid(True, color='grey', linestyle='--')

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('mIoU in \%')
ax2.set_ylabel('ECE in \%')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'.replace('.', ',')))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'.replace('.', ',')))

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

# Increase dpi for higher resolution
plt.savefig('./images/seg_end_iou_ece.png', dpi=300)

# Show the plot
plt.show()