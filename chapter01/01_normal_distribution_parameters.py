import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu=0, sigma=1):
    return 1 / (np.sqrt(2* np.pi) * sigma) * np.exp( - np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

x = np.linspace(-10, 10, 1000)

y0 = normal(x, mu=-3)
y1 = normal(x, mu=0)
y2 = normal(x, mu=5)

plt.plot(x, y0, label='\$mu=-3')
plt.plot(x, y1, label='\$mu=0')
plt.plot(x, y2, label='\$mu=5')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.clf()

y0 = normal(x, sigma=0.5)
y1 = normal(x, sigma=1)
y2 = normal(x, sigma=2)

plt.plot(x, y0, label='\$sigma=0.5')
plt.plot(x, y1, label='\$sigma=1')
plt.plot(x, y2, label='\$sigma=2')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()