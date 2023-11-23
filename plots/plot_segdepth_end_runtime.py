import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size and family
plt.rcParams['font.size'] = '26'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
train_time = [3.919, 4.209, 6.278, 7.134, 8.293, 9.203] # in hours
inf_time = [7.15, 9.20, 22.96, 29.18, 36.36, 41.85] # in milliseconds
inf_std = [1.09, 1.17, 1.02, 1.01, 1.02, 1.15] # in milliseconds

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.10, right=0.90, top=0.97, bottom=0.16)

# Draw points and lines for the first y-axis
ax1.plot(backbones, train_time, marker='o', color='#0072BD', label='Trainingszeit')  # Darker blue

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(backbones, inf_time, marker='o', color='#D95319', label='Inferenzzeit')  # Darker yellow

# Für die linke Y-Achse (ax1)
ax1.tick_params(axis='y', colors='#0072BD')  # Dunkler Blau
ax1.set_ylabel('Trainingszeit in h', color='#0072BD')

# Für die rechte Y-Achse (ax2)
ax2.tick_params(axis='y', colors='#D95319')  # Dunkler Gelb
ax2.set_ylabel('Inferenzzeit in s', color='#D95319')

# Increase the lower and upper limit of the y-axis for the first plot
ax1.set_ylim(bottom=min(train_time) - 1, top=max(train_time) + 0.3)

# Increase the upper limit of the y-axis
ax2.set_ylim(bottom=min(inf_time) - 1, top=max(inf_time) + 10) 

# Add Grid
# Für die linke Y-Achse (ax1)
ax1.grid(True, which='both', color='#0072BD', linestyle='--')

# Für die rechte Y-Achse (ax2)
ax2.grid(True, which='both', color='#D95319', linestyle='--')

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('Trainingszeit in h')
ax2.set_ylabel('Inferenzzeit in ms')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

# Increase dpi for higher resolution
plt.savefig('./images/segdepth_end_runtime.png', dpi=300)

# Show the plot
plt.show()