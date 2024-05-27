import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
    except:
        print('Usage: python 03_sample_sum.py <N>')
        sys.exit(1)
else:
    N = 1

X_sums = []


for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand()
        xs.append(x)
        
    t = np.sum(xs)
    X_sums.append(t)

def normal(x, mu=0, sigma=1):
    return 1 / (np.sqrt(2* np.pi) * sigma) * np.exp( - np.power(x - mu, 2) / (2 * np.power(sigma, 2)))
x_norm = np.linspace(min(X_sums), max(X_sums), 1000)
mu = N / 2
sigma = np.sqrt(N / 12)
y_norm = normal(x_norm, mu, sigma)

plt.hist(X_sums, bins='auto', density=True)
plt.plot(x_norm, y_norm)
plt.title(f'N={N}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.show()