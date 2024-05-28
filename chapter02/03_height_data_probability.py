import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

xs = np.loadtxt('height.txt')

mu = np.mean(xs)
sigma = np.std(xs)

p1 = norm.cdf(160, mu, sigma)
print('p(x <= 160):', p1)

p2 = norm.cdf(180, mu, sigma)
print('p(x > 180):', 1 - p2)