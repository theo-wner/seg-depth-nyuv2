import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set the font size and family
plt.rcParams['font.size'] = '22'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
lrs = ['$1 \cdot 10^{-5}$', '$5 \cdot 10^{-5}$', '$1 \cdot 10^{-4}$']
rmse = np.array([0.5416, 0.5516, 0.5617])

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.14, right=0.86, top=0.97, bottom=0.14)

# Draw points and lines for the first y-axis
ax1.plot(lrs, rmse, marker='o', color='#0072BD', label='mIoU')  # Darker blue

# Für die linke Y-Achse (ax1)
ax1.set_ylabel('RMSE in m')

# Add Grid
ax1.grid(True, color='grey', linestyle='--')

# Add labels to the axes
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('RMSE in m')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'))

# Increase dpi for higher resolution
plt.savefig('./images/exp3_depth_end_rmse.png', dpi=300)

# Show the plot
plt.show()