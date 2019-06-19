import matplotlib.pyplot as plt
import numpy as np
from qgmm_dist import QGMMDist

log2pi = 1.83787706640934533908193770912475883

def pdf(x, mean, cov):
  k = len(mean)
  k_log_2pi = k * log2pi
  
  x = np.reshape(x, (k, 1))

  mean = mean * np.ones((k, 1))
  
  cov = cov * np.eye(k)
  
  log_det_cov = np.log(np.linalg.det(cov))

  inv_cov = np.linalg.inv(cov)

  diffs = x - mean
  
  maha = np.dot(np.dot(np.transpose(diffs), inv_cov), diffs)
  
  log_pdf = -0.25 * (k_log_2pi + log_det_cov + maha)
  
  return np.exp(log_pdf)

def gmm(x, Dist1, Dist2, weights):
  G1 = pdf(x, Dist1.mean, Dist1.cov)
  G1_square = G1 * G1

  G2 = pdf(x, Dist2.mean, Dist2.cov)
  G2_square = G2 * G2

  cos = (1 - weights[0] * weights[0] - weights[1] * weights[1]) / 2 * weights[0] * weights[1] * G1 * G2

  p1 = weights[0] * weights[0] * G1_square + weights[0] * weights[1] * G1 * G2 * cos

  p2 = weights[1] * weights[1] * G2_square + weights[0] * weights[1] * G1 * G2 * cos

  return p1 + p2

def EM(x, Dist1, Dist2, weights):
  # Calculate Q, the equation 10 in the paper
  k = len(Dist1.mean)

  G1 = pdf(x, Dist1.mean, Dist1.cov)
  G2 = pdf(x, Dist2.mean, Dist2.cov)

  o = (G1 * G2) / np.sum(G1 * G2)

  ao = (1 - weights[0] * weights[0] - weights[1] * weights[1]) * o

  Q_norm_const = weights[0] * weights[0] * G1 * G1 + weights[1] * weights[1] * G2 * G2 + ao
  Q1 = (weights[0] * weights[0] * G1 * G1 + (1/2) * ao) / Q_norm_const
  Q2 = (weights[1] * weights[1] * G2 * G2 + (1/2) * ao) / Q_norm_const

  F1 = Q1 - o * np.sum(((1/2) * ao) / Q_norm_const)
  F2 = Q2 - o * np.sum(((1/2) * ao) / Q_norm_const)

  mean1 = (F1 * x) / np.sum(F1)
  mean2 = (F2 * x) / np.sum(F2)

  R1 = 2 * F1
  R2 = 2 * F2

  C1 = np.sum(R1 * (x - mean1) * np.transpose((x - mean1))) / np.sum(R1)
  C2 = np.sum(R2 * (x - mean1) * np.transpose((x - mean1))) / np.sum(R2)

  weights[0] = np.sum(Q1) / k
  weights[1] = np.sum(Q2) / k

  print(Q1)
  print(Q2)

  print(mean1)
  print(mean2)

  print(C1)
  print(C2)

  print(weights[0])
  print(weights[1])
    

# Set means and covariances
mean1 = [-5]
cov1 = [5]
mean2 = [5]
cov2 = [10]

# Create QGMMDists 
Dist1 = QGMMDist(mean1, cov1)
Dist2 = QGMMDist(mean2, cov2)

# Set weights
weights = [.7, .3]
weights = np.sqrt(weights)

# Generate data
xs = np.linspace(-20, 20, 1000)
ys = []
for x in xs:
  y = gmm(x, Dist1, Dist2, weights)
  ys.append(y[0])

EM(x, Dist1, Dist2, weights)

# Plot
#fig = plt.figure()
#plt.plot(xs, ys, 'k')
#plt.show()