import os
import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('height_weight.txt')

print(xs.shape)


small_xs  = xs[:500]

plt.scatter(small_xs[:, 0], small_xs[:, 1])
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()

mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False)

# xは一次元配列か列ベクトル
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp(-0.5 * (x - mu).T @ inv @ (x- mu))
    return y

fig = plt.figure()

ax1 = fig.add_subplot(121, projection='3d')
x_range = np.linspace(xs[:, 0].min(), xs[:, 0].max(), 100)
y_range = np.linspace(xs[: ,1].min(), xs[: ,1].max(), 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z)
ax2.scatter(xs[:, 0], xs[:, 1], alpha=0.3, s=0.1)
plt.show()