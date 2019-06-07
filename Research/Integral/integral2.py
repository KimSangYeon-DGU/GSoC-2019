import math
import numpy as np
from scipy.integrate import quad

d = 2

mean1 = np.array([0, 1])

cov1 = np.matrix([[1, 0],
                 [0, 1]])

det_cov1 = np.linalg.det(cov1)

inv_cov1 = np.linalg.inv(cov1)

'''
def pdf(x, mean, cov):
  diff = x - mean
  return (1 / math.sqrt(math.pow((2 * math.pi), d) * det_cov1)) * math.exp((-1/2) * np.dot(np.dot(diff, inv_cov1), np.transpose(diff)))
'''
log2pi = 1.83787706640934533908193770912475883

def pdf(x, mean, cov):
  return math.exp((-d/2) * log2pi + ((-1/2) * np.log(det_cov1)) + ((-1/2) * np.dot(np.dot(x - mean1, inv_cov1), np.transpose(x - mean1))))

#print(pdf(x, mean1, cov1))
#a = np.transpose(x - mean1)
#print(a.shape)
#diff = x - mean1
#print(inv_cov1)
#print(np.dot(np.dot(diff, inv_cov1), np.transpose(diff)))
#print(pdf(x, mean1, cov1))
I = quad(pdf, -np.Inf, np.Inf, args=(mean1, cov1))
print(I)