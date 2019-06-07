import math
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

d = 1

mean1 = np.array([0])

cov1 = np.matrix([100])

det_cov1 = np.linalg.det(cov1)

inv_cov1 = np.linalg.inv(cov1)

'''
def pdf(x, mean, cov):
  diff = x - mean
  return (1 / math.sqrt(math.pow((2 * math.pi), d) * det_cov1)) * math.exp((-1/2) * np.dot(np.dot(diff, inv_cov1), np.transpose(diff)))
'''

log2pi = 1.83787706640934533908193770912475883

def pdf(x):
  return math.exp( (-1/2) * (d * log2pi + math.log(det_cov1) + np.dot(np.dot(x - mean1, inv_cov1), np.transpose(x - mean1))))

#print(pdf(x, mean1, cov1))
#a = np.transpose(x - mean1)
#print(a.shape)
#diff = x - mean1
#print(inv_cov1)
#print(np.dot(np.dot(diff, inv_cov1), np.transpose(diff)))
#print(pdf(x, mean1, cov1))

x = np.linspace(-20, 20, 100)

y = []

for i in x:
  y.append(pdf(i))

plt.plot(x,y, color='black')

I = quad(pdf, -15, 15)
print(I)

plt.show()