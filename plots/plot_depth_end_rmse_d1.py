import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define a function to change decimal points to commas
def format_ticks(value, tick_number):
    return "%.2f" % value.replace('.', ',')

# Set the font size and family
plt.rcParams['font.size'] = '18'
plt.rcParams['font.family'] = 'serif'

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
val_rmse = [0.6479, 0.6168, 0.5693, 0.5516, 0.5462, 0.5379]
val_d1 = [0.7317, 0.7588, 0.7984, 0.8130, 0.8179, 0.8236]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.13, right=0.87, top=0.97, bottom=0.13)

# Draw points and lines for the first y-axis
ax1.plot(backbones, val_rmse, marker='o', color='#0072BD', label='RMSE')  # Darker blue

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(backbones, val_d1, marker='o', color='#EDB120', label='d1')  # Darker yellow

# Extend the right limit of the x-axis
#ax1.set_xlim(-0.3, len(backbones) - 0.5 + 0.4)  # Add 0.2 to the right limit

# Increase the lower and upper limit of the y-axis for the first plot
#ax1.set_ylim(bottom=min(val_iou) - 1.15, top=max(val_iou) + 0.7)  # Subtract 1 from the lower limit and add 1 to the upper limit

# Increase the upper limit of the y-axis
#ax2.set_ylim(bottom=min(val_calibration_error) - 0.2, top=max(val_calibration_error) + 0.3)  # Add 1 to the upper limit

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('RMSE in m')
ax2.set_ylabel('d1 in ?')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

# Add values to the points
for i, txt in enumerate(val_rmse):
    ax1.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], val_rmse[i]), color='k')  # Subtract 0.7 from the y-coordinate

for i, txt in enumerate(val_d1):
    if i == 0:  # Change the position of the first annotation
        ax2.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], val_d1[i]), color='k')  # Subtract 0.2 from the y-coordinate
    else:
        ax2.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], val_d1[i]), color='k')  # Add 0.1 to the y-coordinate

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

# Increase dpi for higher resolution
plt.savefig('depth_end_rmse_d1.png', dpi=300)

# Show the plot
plt.show()