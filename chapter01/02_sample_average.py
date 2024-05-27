import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
    except:
        print('Usage: python 02_sample_average.py <N>')
        sys.exit(1)
else:
    N = 1

X_means = []


for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand()
        xs.append(x)
        
    mean = np.mean(xs)
    X_means.append(mean)

plt.hist(X_means, bins='auto', density=True)
plt.title(f'N={N}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(-0.05, 1.05)
plt.show()