import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import numpy as np
from Utils import *
from scipy.integrate import dblquad
import math

def pdf(x, mean, cov):
    dim, mean, cov = process_parameters(None, mean, cov)
    x = process_quantiles(x, dim)
    prec_U, log_det_cov = psd_pinv_decomposed_log_pdet(cov)
    out = np.exp(logpdf(x, mean, prec_U, log_det_cov))

    return np.sqrt(squeeze_output(out))

def getCosine(G1, G2, weights):
  return np.cos((1 - (weights[0] * weights[0]) - (weights[1] * weights[1])) / (2.0 * weights[0] * weights[1] * G1 * G2))
  
def QuantumGMM(G1, G2, weights):
  cosine = getCosine(G1, G2, weights)
  print(np.min(cosine), np.max(cosine)) # -1 <= cos <= 1

  P1 = (weights[0] * weights[0]) * (G1 * G1) + weights[0] * weights[1] * G1 * G2 * cosine
  P2 = (weights[1] * weights[1]) * (G2 * G2) + weights[0] * weights[1] * G1 * G2 * cosine

  return P1 + P2

def getVolume(X, Y, Z):
  v = X[0] * Y[0] * Z[0]
  dim = X.shape[0]
  for j in range(1, dim):
    for i in range(1, dim):
      v += (X[j][i] - X[j][i-1]) * (Y[j][i] - Y[j-1][i]) * Z[j][i]

  return np.sum(v)

fig = plt.figure()
ax = fig.gca(projection='3d')
jet = plt.get_cmap('jet')

d = 2 # Number of dimensions

# Mean and covariance of the first Gaussian distribution
mean1 = [-2, -1]
cov1 = [2, 3]

# Mean and covariance of the second Gaussian distribution
mean2 = [3, 2]
cov2 = [2, 1]

# Weight of each Gaussian distribution
w = [math.sqrt(0.6), math.sqrt(0.4)]

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)

X, Y = np.meshgrid(x, y)

# observations
obs = np.empty(X.shape + (2, ))

obs[:, :, 0] = X; obs[:, :, 1] = Y

# Generate Gaussian distributions according to the parameters
G1 = pdf(obs, mean1, cov1)

# Calculate volume of G1
v1 = getVolume(X, Y, G1)
print("The volume of the first Gaussian: {0}".format(v1))

# Calculate volume of G2
G2 = pdf(obs, mean2, cov2)
v2 = getVolume(X, Y, G2)
print("The volume of the second Gaussian: {0}".format(v2))

# Gaussian Mixture
G_mix = QuantumGMM(G1, G2, w)
v3 = getVolume(X, Y, G_mix)
print("The volume of the Gaussian mixture: {0}".format(v3))

surf = ax.plot_surface(X, Y, G_mix, cmap=jet,linewidth=0)
fig.savefig("SaveQGMM.png")
plt.show()