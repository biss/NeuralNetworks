import numpy as np
import matplotlib.pyplot as plt

def cost_per_epoch(cost, label):
    num_epochs = len(cost)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, num_epochs), cost[:num_epochs], color='#2A6EA6')
    ax.set_xlim([0, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(label)
    plt.show()

