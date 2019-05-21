import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import numpy as np
from Utils import *

def pdf(x, mean, cov):
    dim, mean, cov = process_parameters(None, mean, cov)
    x = process_quantiles(x, dim)
    prec_U, log_det_cov = psd_pinv_decomposed_log_pdet(cov)
    out = np.exp(logpdf(x, mean, prec_U, log_det_cov))

    return squeeze_output(out)

fig = plt.figure()
ax = fig.gca(projection='3d')
jet = plt.get_cmap('jet')

d = 2 # Number of dimensions

# Mean and covariance of the first Gaussian distribution
mean1 = [-3, -1]
cov1 = [2, 3]

# Mean and covariance of the second Gaussian distribution
mean2 = [2, 1]
cov2 = [4, 3]

# Weight of each Gaussian distribution
w = [0.6, 0.4]

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)

X, Y = np.meshgrid(x, y)

# observations
obs = np.empty(X.shape + (2, ))

obs[:, :, 0] = X; obs[:, :, 1] = Y

# Generate Gaussian distributions according to the parameters
G1 = pdf(obs, mean1, cov1)
G2 = pdf(obs, mean2, cov2)

# Gaussian Mixture
G_mix = w[0]*G1 + w[1]*G2

surf = ax.plot_surface(X, Y, G_mix, cmap=jet,linewidth=0)

plt.show()