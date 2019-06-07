import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import math
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
  
  log_pdf = -0.5 * (k_log_2pi + log_det_cov + maha)
  
  return np.exp(log_pdf)

mean = 0
cov = 10
k = 1

I = quad(pdf, -np.Inf, np.Inf, args=(mean, cov, k))
print(I)