import numpy as np

x = np.array([1, 2, 3])

print(x.__class__)
print(x.ndim)
print(x.shape)

print('-'*40)

W = np.array([[1, 2, 3], [4, 5, 6]])

print(W.ndim)
print(W.shape)

print('-'*40)

W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])

print(W + X)
print('---')
print(W - X)

print('-'*40)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = np.dot(a, b)

print(y)
print(a @ b)

print('-'*40)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Y = np.dot(A, B)

print(Y)
print(A @ B)