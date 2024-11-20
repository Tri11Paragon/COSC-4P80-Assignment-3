import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import sys

filename = sys.argv[1]
size = sys.argv[2]

df = pd.read_csv(filename, header=None)

height, width = df.shape

data = df.to_numpy()

plt.imshow(data, cmap='coolwarm_r', interpolation='nearest')

plt.xticks(np.arange(width), np.arange(width))
plt.yticks(np.arange(height), np.arange(height))

plt.xlabel('X Pos')
plt.ylabel('Y Pos')
plt.title('Heatmap of Motor Data (Bins: {})'.format(size))

plt.gca().invert_yaxis()

plt.colorbar()

plt.savefig("heatmap.png")
