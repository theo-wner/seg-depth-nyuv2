import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size and family
plt.rcParams['font.size'] = '22'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']
val_rmse = [0.6479, 0.6168, 0.5693, 0.5516, 0.5462, 0.5379]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.12, right=0.88, top=0.97, bottom=0.15)

# Draw points and lines for the first y-axis
ax1.plot(backbones, val_rmse, marker='o', color='#0072BD', label='RMSE')  # Darker blue

# Increase the lower and upper limit of the y-axis for the first plot
ax1.set_ylim(bottom=min(val_rmse) - 0.005, top=max(val_rmse) + 0.010) 

# Add grid lines
ax1.grid(color='grey', linestyle='dashed')

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('RMSE in m')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'.replace('.', ',')))

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()

# Increase dpi for higher resolution
plt.savefig('./images/depth_end_rmse.png', dpi=300)

# Show the plot
plt.show()