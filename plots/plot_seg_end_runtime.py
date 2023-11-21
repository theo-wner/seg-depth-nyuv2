import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size and family
plt.rcParams['font.size'] = '18'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Data
backbones = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
train_time = [2.023, 2.288, 3.704, 4.642, 5.738, 6.663]
inf_time = [1, 1, 1, 1, 1, 1]

# Create a figure and an axis with a wider size
fig, ax1 = plt.subplots(figsize=(8, 5)) 

# Adjust the margins of the figure
fig.subplots_adjust(left=0.13, right=0.87, top=0.97, bottom=0.13)

# Draw points and lines for the first y-axis
ax1.plot(backbones, train_time, marker='o', color='#0072BD', label='Trainingszeit')  # Darker blue

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(backbones, inf_time, marker='o', color='#EDB120', label='Inferenzzeit')  # Darker yellow

# Extend the right limit of the x-axis
ax1.set_xlim(-0.3, len(backbones) - 0.5 + 0.4)  # Add 0.2 to the right limit

# Increase the lower and upper limit of the y-axis for the first plot
#ax1.set_ylim(bottom=min(train_time) - 1.15, top=max(train_time) + 0.7)  # Subtract 1 from the lower limit and add 1 to the upper limit

# Increase the upper limit of the y-axis
#ax2.set_ylim(bottom=min(inf_time) - 0.2, top=max(inf_time) + 0.3)  # Add 1 to the upper limit

# Add labels to the axes
ax1.set_xlabel('Backbone')
ax1.set_ylabel('Trainingszeit in h')
ax2.set_ylabel('Inferenzzeit in s')

# Change the decimal separator to comma
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))

# Convert hours to hh:mm and round seconds
for i, txt in enumerate(train_time):
    hours = int(txt)
    minutes = int((txt*60) % 60)
    ax1.annotate('{:02d}:{:02d}'.format(hours, minutes), (backbones[i], train_time[i]), color='k')  # Subtract 0.7 from the y-coordinate

for i, txt in enumerate(inf_time):
    ax2.annotate('{:.2f}'.format(txt).replace('.', ','), (backbones[i], inf_time[i]), color='k')  # Add 0.1 to the y-coordinate

# Add a legend to the middle right
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')

# Increase dpi for higher resolution
plt.savefig('./images/seg_end_runtime.png', dpi=300)

# Show the plot
plt.show()