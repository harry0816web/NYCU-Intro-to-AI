"""
depth 4:
    alpha    1421658.04 
    minimax 31011299.26
depth 3:
    alpha 298510.72ms
    minimax 634223.76ms

depth 2:
    alpha  51821.55
    minimax 59800.14

depth 1:
    alpha 6875.80
    minimax 7328.32
"""
# draw a comparison of alpha-beta pruning and minimax algorithm with data above
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# data for alpha-beta pruning and minimax algorithm
depth = [1, 2, 3, 4]
alpha = [6875.80, 51821.55, 298510.72, 1421658.04]
minimax = [7328.32, 59800.14, 634223.76, 4568996.30]

# create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
# set the title and labels
ax.set_title('Alpha-Beta Pruning vs Minimax Algorithm', fontsize=16)
ax.set_xlabel('Depth', fontsize=14)
ax.set_ylabel('Time (ms)', fontsize=14)
# set the x and y axis limits
ax.set_xlim(0, 5)
ax.set_ylim(0, 5000000)
# set the x and y axis ticks
ax.set_xticks(depth)
ax.set_yticks(np.arange(0, 310000, 5000000))
# set the x and y axis tick labels
ax.set_xticklabels(depth, fontsize=12)
ax.set_yticklabels(np.arange(0, 310000, 5000000), fontsize=12)
# set the grid
ax.grid(True, linestyle='--', alpha=0.7)
# set the legend
alpha_patch = mpatches.Patch(color='blue', label='Alpha-Beta Pruning')
minimax_patch = mpatches.Patch(color='orange', label='Minimax Algorithm')
ax.legend(handles=[alpha_patch, minimax_patch], fontsize=12)
# plot the data
ax.plot(depth, alpha, color='blue', marker='o', label='Alpha-Beta Pruning')
ax.plot(depth, minimax, color='orange', marker='o', label='Minimax Algorithm')
# add the data labels
for i in range(len(depth)):
    ax.text(depth[i], alpha[i], str(alpha[i]), fontsize=10, ha='center', va='bottom')
    ax.text(depth[i], minimax[i], str(minimax[i]), fontsize=10, ha='center', va='bottom')
# show the plot
plt.tight_layout()
plt.show()
# save the plot
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
# save the plot as pdf
plt.savefig('comparison.pdf', dpi=300, bbox_inches='tight')
