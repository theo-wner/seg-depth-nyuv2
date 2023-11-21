import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size and family
plt.rcParams['font.size'] = '18'
plt.rcParams['font.family'] = 'serif'

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
val_rmse = [0.6789, 0.6253, 0.5657, 0.5457, 0.5316, 0.5262]
val_d1 = [0.7043, 0.7433, 0.7946, 0.8154, 0.8234, 0.8209]

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
ax1.set_xlim(-0.3, len(backbones) + 0.1) 

# Increase the lower and upper limit of the y-axis for the first plot
ax1.set_ylim(bottom=min(val_rmse) - 0.005, top=max(val_rmse) + 0.015) 

# Increase the upper limit of the y-axis
ax2.set_ylim(bottom=min(val_d1) - 0.012, top=max(val_d1) + 0.003) 

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('RMSE in m')
ax2.set_ylabel('d1 in ?')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'.replace('.', ',')))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'.replace('.', ',')))

# Add values to the points
for i, txt in enumerate(val_rmse):
    ax1.annotate('{:.4f}'.format(txt).replace('.', ','), (backbones[i], val_rmse[i] + 0.003), color='k') 

for i, txt in enumerate(val_d1):
    ax2.annotate('{:.4f}'.format(txt).replace('.', ','), (backbones[i], val_d1[i] - 0.008), color='k') 

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

# Increase dpi for higher resolution
plt.savefig('./images/segdepth_end_rmse_d1.png', dpi=300)

# Show the plot
plt.show()