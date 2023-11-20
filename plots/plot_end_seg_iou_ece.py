import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define a function to change decimal points to commas
def comma_decimal(x, pos):
    return str(x).replace('.', ',')

# Set the font size and family
plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'serif'

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
val_iou = [39.64, 42.33, 47.90, 51.07, 51.64, 51.93]
val_calibration_error = [22.92, 23.87, 22.06, 21.17, 20.98, 20.95]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Draw points and lines for the first y-axis
ax1.plot(backbones, val_iou, marker='o', color='#0072BD', label='mIoU')  # Darker blue

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(backbones, val_calibration_error, marker='o', color='#EDB120', label='ECE')  # Darker yellow

# Extend the right limit of the x-axis
ax1.set_xlim(-0.5, len(backbones) - 0.5 + 0.2)  # Add 0.2 to the right limit

# Increase the lower and upper limit of the y-axis for the first plot
ax1.set_ylim(bottom=min(val_iou) - 1.15, top=max(val_iou) + 0.7)  # Subtract 1 from the lower limit and add 1 to the upper limit

# Increase the upper limit of the y-axis
ax2.set_ylim(bottom=min(val_calibration_error) - 0.2, top=max(val_calibration_error) + 0.3)  # Add 1 to the upper limit

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('mIoU in %')
ax2.set_ylabel('ECE in %')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(comma_decimal))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(comma_decimal))

# Add values to the points
for i, txt in enumerate(val_iou):
    ax1.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], val_iou[i] - 0.75), color='k')  # Subtract 0.7 from the y-coordinate

for i, txt in enumerate(val_calibration_error):
    if i == 0:  # Change the position of the first annotation
        ax2.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], val_calibration_error[i] - 0.2), color='k')  # Subtract 0.2 from the y-coordinate
    else:
        ax2.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], val_calibration_error[i] + 0.1), color='k')  # Add 0.1 to the y-coordinate

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

# Increase dpi for higher resolution
plt.savefig('end_seg_iou_ece.png', dpi=300)

# Show the plot
plt.show()