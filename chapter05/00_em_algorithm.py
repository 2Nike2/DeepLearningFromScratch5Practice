import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('old_faithful.txt')
print(xs.shape)

# パラメータ初期値
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)])

K = len(phis)
N = len(xs)
MAX_ITERS = 100
THRESHOLD = 1e-4

# xは一次元配列か列ベクトル
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp(-0.5 * (x - mu).T @ inv @ (x- mu))
    return y

def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y

def likelihood(xs, phis, mus, covs):
    eps = 1e-8
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N

current_likelihood = likelihood(xs, phis, mus, covs)
for iter in range(MAX_ITERS):

    # E-step
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)

    # M-step
    qs_sum = qs.sum(axis=0)
    for k in range(K):

        phis[k] = qs_sum[k] / N

        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[: , np.newaxis]
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]

    print(f'{current_likelihood:.3f}')

    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood)
    if diff < THRESHOLD:
        break
    current_likelihood = next_likelihood

figure, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(xs[:, 0], xs[:, 1], c='blue', marker='o')

x_range = np.linspace(1, 6, 100)
y_range = np.linspace(40, 100, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = gmm(x, phis, mus, covs)

ax.contour(X, Y, Z)

plt.show()

figure, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(xs[:, 0], xs[:, 1], c='blue', alpha=0.5)
sample_num = N
new_xs = np.zeros((sample_num, 2))
for n in range(sample_num):
    k = np.random.choice(K, p=phis)
    mu, cov = mus[k], covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)
ax.scatter(new_xs[:, 0], new_xs[:, 1], c='orange', alpha=0.5)

plt.show()