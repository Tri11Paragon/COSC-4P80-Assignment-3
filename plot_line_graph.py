import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

file1 = sys.argv[1]
file2 = sys.argv[2]
bins = sys.argv[3]
split = sys.argv[4]
if len(sys.argv) > 4:
    subtitle1 = sys.argv[5]
    subtitle2 = sys.argv[6]
else:
    subtitle1 = ""
    subtitle2 = ""

print(file1)
print(file2)
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
print(dir_path)
print(cwd)

with open(file1, "r+") as f:
    print(f.name)

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

data1 = df1.to_numpy()
data2 = df2.to_numpy()

y_min = np.min(data2)
y_max = np.max(data2)

if split.lower() == "false":
    fig, ax1 = plt.subplots()

    ax1.plot(data1, color='b', label='Topological Error')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error %', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 1)
    #ax1.set_xlim(0, data1.size)

    ax2 = ax1.twinx()

    ax2.plot(data2, color='r', label='Quantization Error')
    ax2.set_ylabel('Incorrect BMU', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(y_min, y_max)

    ax1.set_title('Topological and Quantization Error (Bins: {})'.format(bins))
    fig.suptitle(subtitle1)

    plt.savefig("errors{}.png".format(bins))
else:
    plt.plot(data1, color='b', label='Topological Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error %', color='b')
    plt.tick_params(axis='y', labelcolor='b')
    plt.ylim(0, 1)
    
    plt.suptitle("Topological Error (Bins: {})".format(bins))
    plt.title(subtitle1)

    plt.savefig("errors-topological{}.png".format(bins))
    
    plt.plot(data2, color='b', label='Quantization Error')
    plt.xlabel('Epoch')
    plt.ylabel('Incorrect BMU', color='b')
    plt.tick_params(axis='y', labelcolor='b')
    plt.ylim(y_min, y_max)
    
    plt.suptitle("Quantization Error (Bins: {})".format(bins))
    plt.title(subtitle2)

    plt.savefig("errors-quantization{}.png".format(bins))


