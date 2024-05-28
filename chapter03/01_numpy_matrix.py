import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print('---')
print(A.T)

print('-'*40)

A = np.array([[3, 4], [5, 6]])
d = np.linalg.det(A)
print(d)

print('-'*40)

A = np.array([[3, 4], [5, 6]])
B = np.linalg.inv(A)

print(B)
print('---')
print(A @ B)
print(B @ A)

print('-'*40)

# xは一次元配列か列ベクトル
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp(-0.5 * (x - mu).T @ inv @ (x- mu))
    return y

x = np.array([[0], [0]])
mu = np.array([[1], [2]])
cov = np.array([[1, 0], [0, 1]])

y = multivariate_normal(x, mu, cov)

print(y)