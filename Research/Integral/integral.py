import numpy as np
import math

d = 2

# Gaussian integral
mean1 = [0, 0]
cov1 = [[2, 1],
              [1, 1]]

inv_cov1 = np.linalg.inv(cov1)
det_cov1 = np.linalg.det(cov1)
det_inv_cov1 = np.linalg.det(inv_cov1)

I = math.sqrt(math.pow((2 * math.pi), d) / det_inv_cov1)

norm_const = 1 / math.sqrt(math.pow(2 * math.pi, d) * det_cov1)

print(I)

print(norm_const * I)


# Quantum Gaussian integral
alpha1 = math.sqrt(0.6)
alpha2 = math.sqrt(0.4)

mean2 = [0, 0]
cov2 = [[3, 1],
        [1, 1]]

inv_cov2 = np.linalg.inv(cov2)

det_inv_cov2 = np.linalg.det(inv_cov2)

I1 = math.sqrt(math.sqrt(math.pow((2 * math.pi), d) / det_inv_cov1))
I2 = math.sqrt(math.sqrt(math.pow((2 * math.pi), d) / det_inv_cov2))

print(I1 * I2)

norm = 2 * alpha1 * alpha2

norm2 = 

print(norm * I1 * I2 + alpha1 * alpha1 + alpha2 * alpha2)
print(norm * I1 * I2 + alpha1 * alpha1 + alpha2 * alpha2)

'''
nc = ((1 - (alpha1 * alpha1) - (alpha2 * alpha2)) / 2 * alpha1 * alpha2)

nc = nc * I1 * I2
print(nc * I1 * I2)
'''
