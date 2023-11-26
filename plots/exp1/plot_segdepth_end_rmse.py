import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size and family
plt.rcParams['font.size'] = '26'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
val_rmse = [0.6789, 0.6253, 0.5657, 0.5457, 0.5316, 0.5262]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.14, right=0.86, top=0.97, bottom=0.16)

# Draw points and lines for the first y-axis
ax1.plot(backbones, val_rmse, marker='o', color='#0072BD', label='RMSE')  # Darker blue

# Increase the lower and upper limit of the y-axis for the first plot
ax1.set_ylim(bottom=min(val_rmse) - 0.005, top=max(val_rmse) + 0.015) 

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
plt.savefig('./images/segdepth_end_rmse.png', dpi=300)

# Show the plot
plt.show()