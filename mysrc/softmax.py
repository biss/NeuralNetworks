"""Softmax."""
import numpy as np

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]],dtype=float)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if type(x) == list:
        dim=len(x)
        norm = np.sum(np.exp(x))
        for idx in range(dim):
            x[idx] = np.exp(x[idx])/norm
    elif type(x) == np.ndarray:
        dim=x.shape
        for col in range(dim[1]):
            norm = np.sum(np.exp(x[:, col]))
            for idx in range(dim[0]):
                x[idx, col] = np.exp(x[idx, col])/norm
    else:
        raise Exception('incorrect input')
    return x

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
