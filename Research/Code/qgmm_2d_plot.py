import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

log2pi = 1.83787706640934533908193770912475883

def pdf(x, mean, cov, k):
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

def gmm(x, mean1, cov1, mean2, cov2, weights, k):
  g1 = pdf(x, mean1, cov1, k)
  g1_sqr = g1 * g1
  g2 = pdf(x, mean2, cov2, k)
  g2_sqr = g2 * g2

  cos = (1 - weights[0] * weights[0] - weights[1] * weights[1]) / 2 * weights[0] * weights[1] * g1 * g2

  p1 = weights[0] * weights[0] * g1_sqr + weights[0] * weights[1] * g1 * g2 * cos

  p2 = weights[1] * weights[1] * g2_sqr + weights[0] * weights[1] * g1 * g2 * cos

  return p1 + p2

def check(x, mean1, cov1, mean2, cov2, weights, k):
  g1 = pdf(x, mean1, cov1, k)
  
  g2 = pdf(x, mean2, cov2, k)

  cos = (1 - weights[0] * weights[0] - weights[1] * weights[1]) / 2 * weights[0] * weights[1] * g1 * g2
  
  return cos

mean1 = -5
cov1 = 5

mean2 = 5
cov2 = 10

k = 1
weights = [.7, .3]
weights = np.sqrt(weights)

xs = np.linspace(-20, 20, 1000)
ys = []
for x in xs:
  y = gmm(x, mean1, cov1, mean2, cov2, weights, k)
  #y = pdf(x, mean1, cov1, k)
  #y = pdf(x, mean2, cov2, k)
  ys.append(y[0])

fig = plt.figure()

ys = np.asarray(ys)
print(ys.shape)
plt.plot(xs, ys, 'g')
fig.savefig("QGMM.png")