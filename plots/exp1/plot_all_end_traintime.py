import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set the font size and family
plt.rcParams['font.size'] = '22'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']
train_time_seg = [2.023, 2.288, 3.704, 4.642, 5.738, 6.663] # in hours
inf_time_seg = [6.75, 7.76, 17.05, 23.28, 30.36, 37.62] # in milliseconds
train_time_depth = [3.158, 3.442, 4.889, 5.785, 6.941, 7.879] # in hours
inf_time_depth = [6.49, 7.31, 16.62, 22.81, 29.96, 37.35] # in milliseconds
train_time_segdepth = [3.919, 4.209, 6.278, 7.134, 8.293, 9.203] # in hours
inf_time_segdepth = [7.15, 9.20, 22.96, 29.18, 36.36, 41.85] # in milliseconds
train_time_segplusdepth = np.array(train_time_seg) + np.array(train_time_depth)
inf_time_segplusdepth = np.array(inf_time_seg) + np.array(inf_time_depth)

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.10, right=0.90, top=0.97, bottom=0.15)

# Draw points and lines for the first y-axis
ax1.plot(backbones, train_time_seg, marker='o', color='#0072BD', label='Segmentierung')  # Darker blue
ax1.plot(backbones, train_time_depth, marker='o', color='#D95319', label='Tiefenschätzung')  # Darker orange
ax1.plot(backbones, train_time_segdepth, marker='o', color='#EDB120', label='Simultan')  # Darker yellow
ax1.plot(backbones, train_time_segplusdepth, marker='o', color='#7E2F8E', label='Summiert')  # Darker pursple

# Make sure y axis starts at 0
ax1.set_ylim(bottom=0)

# Add Grid
ax1.grid(True, color='grey', linestyle='--')

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('Trainingszeit in h')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='upper left', ncol=1)

# Increase dpi for higher resolution
plt.savefig('./images/all_end_traintime.png', dpi=300)

# Show the plot
plt.show()