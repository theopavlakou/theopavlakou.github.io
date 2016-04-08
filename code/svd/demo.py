import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.linalg as lin
import os
from os.path import join

cwd = os.getcwd()
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(15, 15, forward=True)

x_min = -3
x_max = 3
y_min = -3
y_max = 3

# The transformation matrix
A = 2*np.array([  [np.sqrt(3)/2, 0.5],
                [-0.5, np.sqrt(3)/2]])

# The SVD of the transformation matrix
U, S, V_T = lin.svd(A)
S = np.diag(S)

print(U)
print(S)
print(V_T)

# X = np.array([  [0, 0],
#                 [np.sqrt(2)/2, np.sqrt(2)/2],
#                 [np.sqrt(2)/2, -np.sqrt(2)/2],
#                 [0, 0]]).T

X = np.array([  [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 0]]).T

X_orginal = np.eye(2).dot(X)

ax[0, 0].plot(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[0, 0].fill(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[0, 0].set_xlim([x_min, x_max])
ax[0, 0].set_ylim([y_min, y_max])
ax[0, 0].set_title("Original $x$", size=20)

X = V_T.dot(X)
ax[0, 1].plot(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[0, 1].fill(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[0, 1].set_xlim([x_min, x_max])
ax[0, 1].set_ylim([y_min, y_max])
ax[0, 1].set_title("Original $V^T x$", size=20)

X = S.dot(X)
ax[1, 0].plot(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[1, 0].fill(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[1, 0].set_xlim([x_min, x_max])
ax[1, 0].set_ylim([y_min, y_max])
ax[1, 0].set_title("Original $S V^T x$", size=20)

X = U.dot(X)
X = A.dot(X_orginal)
ax[1, 1].plot(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[1, 1].fill(X[0, :], X[1, :], color=sns.cubehelix_palette(8)[2])
ax[1, 1].set_xlim([x_min, x_max])
ax[1, 1].set_ylim([y_min, y_max])
ax[1, 1].set_title("Original $Ax = U S V^T x$", size=20)

fig.savefig(join(cwd, "img/blog/svd/demo.jpg"))
plt.show()
