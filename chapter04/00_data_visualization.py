import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('old_faithful.txt')

print(xs.shape)
print(xs[0])

plt.scatter(xs[:, 0], xs[:, 1])
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()