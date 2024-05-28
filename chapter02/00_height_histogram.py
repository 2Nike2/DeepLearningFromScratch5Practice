import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('height.txt')
print(xs.shape)

plt.hist(xs, bins='auto', density=True)
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.show()