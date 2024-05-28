import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('height.txt')
print(xs.shape)

mu = np.mean(xs)
sigma = np.std(xs)

print('Mean:', mu)
print('Standard deviation:', sigma)

def normal(x, mu=0, sigma=1):
    return 1 /(np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

x = np.linspace(150, 190, 1000)
y = normal(x, mu, sigma)

plt.hist(xs, bins='auto', density=True)
plt.plot(x, y)
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.show()