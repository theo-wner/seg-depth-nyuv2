import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size and family
plt.rcParams['font.size'] = '26'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
val_iou = [39.34, 41.98, 48.10, 51.66, 51.44, 52.38]
val_calibration_error = [23.20, 24.08, 21.88, 20.94, 20.99, 20.95]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.14, right=0.86, top=0.97, bottom=0.16)

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

# Increase the lower and upper limit of the y-axis for the first plot
ax1.set_ylim(bottom=min(val_iou) - 1.15, top=max(val_iou) + 0.7)  # Subtract 1 from the lower limit and add 1 to the upper limit

# Increase the upper limit of the y-axis
ax2.set_ylim(bottom=min(val_calibration_error) - 0.2, top=max(val_calibration_error) + 0.3)  # Add 1 to the upper limit

# Add Grid
# Für die linke Y-Achse (ax1)
ax1.grid(True, which='both', color='#0072BD', linestyle='--')
# Für die rechte Y-Achse (ax2)
ax2.grid(True, which='both', color='#D95319', linestyle='--')
# Für vertikale Linien
ax1.grid(True, which='major', axis='x', color='grey', linestyle='--')

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
plt.savefig('./images/segdepth_end_iou_ece.png', dpi=300)

# Show the plot
plt.show()