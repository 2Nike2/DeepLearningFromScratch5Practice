import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('height.txt')

mu = np.mean(xs)
sigma = np.std(xs)

samples = np.random.normal(mu, sigma, 10000)

plt.hist(xs, bins='auto', density=True, alpha=0.5, label='original')
plt.hist(samples, bins='auto', density=True, alpha=0.5, label='generated')
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()