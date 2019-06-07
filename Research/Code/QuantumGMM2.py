import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import numpy as np
from Utils import *
from scipy.integrate import dblquad
import math
import random

def pdf(x, mean, cov):
    dim, mean, cov = process_parameters(None, mean, cov)
    x = process_quantiles(x, dim)
    prec_U, log_det_cov = psd_pinv_decomposed_log_pdet(cov)
    out = np.exp(logpdf(x, mean, prec_U, log_det_cov))

    return np.sqrt(squeeze_output(out))

def getCosine(G1, G2, weights):
  return (1 - (weights[0] * weights[0]) - (weights[1] * weights[1])) / (2.0 * weights[0] * weights[1] * G1 * G2)
  
def QuantumGMM(G1, G2, weights, phi):
  #cosine = getCosine(G1, G2, weights)
  cosine = np.ones((500,500))
  '''
  for i in range(500):
    for j in range(500):
      cosine[i, j] = random.uniform(0, math.pi)
  
  cosine *= phi
  cosine = np.cos(np.rad2deg(cosine))
  '''

  cosine = np.ones((500, 500))
  cosine *= phi
  cosine = np.cos(np.deg2rad(cosine))

  print(np.min(cosine), np.max(cosine))
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

d = 2 # Number of dimensions

# Mean and covariance of the first Gaussian distribution
mean1 = [-2, -1]
cov1 = [2, 3]

# Mean and covariance of the second Gaussian distribution
mean2 = [3, 2]
cov2 = [2, 1]

# Weight of each Gaussian distribution

#w = [math.sqrt(0.6), math.sqrt(0.4)]
w1 = random.uniform(0, 1)
w2 = random.uniform(0, 1)
w = [math.sqrt(0.3), math.sqrt(0.22)]
#print(w1, w2)

x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)

X, Y = np.meshgrid(x, y)

# observations
obs = np.empty(X.shape + (2, ))

obs[:, :, 0] = X; obs[:, :, 1] = Y

# Generate Gaussian distributions according to the parameters
G1 = pdf(obs, mean1, cov1)

# Calculate volume of G1
#v1 = getVolume(X, Y, G1)
#print("The volume of the first Gaussian: {0}".format(v1))

# Calculate volume of G2
G2 = pdf(obs, mean2, cov2)
#v2 = getVolume(X, Y, G2)
#print("The volume of the second Gaussian: {0}".format(v2))

# Gaussian Mixture
phis = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

for phi in phis:
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  jet = plt.get_cmap('jet')
  G_mix = QuantumGMM(G1, G2, w, phi)
  surf = ax.plot_surface(X, Y, G_mix, cmap=jet,linewidth=0)
  fig.savefig("Q_phi_{0}".format(phi))


#v3 = getVolume(X, Y, G_mix)
#print("The volume of the Gaussian mixture: {0}".format(v3))

#surf = ax.plot_surface(X, Y, G_mix, cmap=jet,linewidth=0)

#plt.show()