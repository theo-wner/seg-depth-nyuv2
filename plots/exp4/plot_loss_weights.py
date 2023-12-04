import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# Set the font size and family
plt.rcParams['font.size'] = '22'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
iou = np.array([0.5089, 0.5158, 0.5190, 0.5166, 0.5174, 0.5122, 0.5149]) * 100
rmse = np.array([0.5455, 0.5451, 0.5466, 0.5457, 0.5507, 0.5557, 0.5550])

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(9, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.12, right=0.82, top=0.97, bottom=0.15)

# Draw Data points for each tuple of iou and rmse with different colors
colors = np.arange(len(iou))
scatter = ax1.scatter(iou, rmse, c=colors, cmap='plasma_r', marker='o', s=100)  # Darker blue

# Add a colorbar
cbar = plt.colorbar(scatter, ticks=range(len(iou)))

# Set the labels for the colorbar
labels = [
    r'$\alpha_{\mathrm{s}}\!=\!0,7\,;\,\alpha_{\mathrm{t}}\!=\!1,3$',
    r'$\alpha_{\mathrm{s}}\!=\!0,8\,;\,\alpha_{\mathrm{t}}\!=\!1,2$',
    r'$\alpha_{\mathrm{s}}\!=\!0,9\,;\,\alpha_{\mathrm{t}}\!=\!1,1$',
    r'$\alpha_{\mathrm{s}}\!=\!1,0\,;\,\alpha_{\mathrm{t}}\!=\!1,0$',
    r'$\alpha_{\mathrm{s}}\!=\!1,1\,;\,\alpha_{\mathrm{t}}\!=\!0,9$',
    r'$\alpha_{\mathrm{s}}\!=\!1,2\,;\,\alpha_{\mathrm{t}}\!=\!0,8$',
    r'$\alpha_{\mathrm{s}}\!=\!1,3\,;\,\alpha_{\mathrm{t}}\!=\!0,7$'
]
cbar.set_ticklabels(labels)

# Add Grid
ax1.grid(True, color='grey', linestyle='--')

# Add labels to the axes
ax1.set_xlabel('mIoU in \%')
ax1.set_ylabel('RMSE in m')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.3f}'))
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))

# Increase dpi for higher resolution
plt.savefig('./images/loss_weights.png', dpi=300)

# Show the plot
plt.show()