from qgmm_utils import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from matplotlib.patches import Ellipse

df = pd.read_csv('faithful.csv', sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
#print(dataset.T.shape)
CLUSTERS = 2

def eigsorted(cov):
    '''
    Eigenvalues and eigenvectors of the covariance matrix.
    '''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(points, cov, nstd):
    """
    Source: http://stackoverflow.com/a/12321306/1391441
    """

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    return width, height, theta

def plot_clustered_data(points, c_means, covs):
    """Plots the cluster-colored data and the cluster means"""
    colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))
    #ax = plt.gca()
    for i in range(points.shape[1]):
      plt.plot(points[:, i][0], points[:, i][1], ".", color="red", zorder=0)
    
    plt.plot(c_means[0][0], c_means[0][1], ".", color="green", zorder=1)
    plt.plot(c_means[1][0], c_means[1][1], ".", color="blue", zorder=1)

    #width1, height1, theta1 = cov_ellipse(points, covs[0], nstd=0.3)
    #ellipse1 = Ellipse(xy=(c_means[0][0], c_means[0][1]), width=width1, height=height1, angle=theta1,
    #                   edgecolor='b', fc='None', lw=2, zorder=4)
    #ax.add_patch(ellipse1)
    plt.show()

# Set the number of components is 2.
num_components = 2

alphas = tf.Variable([0.3, 0.7], dtype=tf.float32)
means = tf.Variable([[1.67, 50.89705882], [4.88778309, 80.89705882]], dtype=tf.float32)
covs = tf.Variable([[[8, 0], [0, 8]],[[10, 0],[0, 10]]], dtype=tf.float32)

covs_lower = tf.linalg.cholesky(covs)

#obs = tf.constant([[5, 6, 12, 4, 5],
#                   [-2, -2, -3, -2, -1]], dtype=tf.float32)

obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

'''
# Build session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(factor_covariance(covs[0])))
'''
G = []
for i in range(num_components):
  #cov = tf.matmul(tf.transpose(covs_lower[i]), covs_lower[i])
  G.append(unnormalized_gaussians(obs, means[i], covs[i]))

G = tf.convert_to_tensor(G, dtype=tf.float32)
Q = get_Q(G, alphas, num_components)

# lambda
ld = 1

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0
J = tf.reduce_sum(Q[0] * tf.math.log(G[0]) + Q[1] * tf.math.log(G[1])) + ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + 2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])) - 1)

# Set optimizer to Adam with learning rate 0.1
optim = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optim.minimize(-J)

# Build session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)

# Set the number of iterations is 2000.
n_iter = 200
print(sess.run(covs))
for i in range(n_iter):
    _ = sess.run(training_op)
    sess.run(alphas)
    sess.run(means)
    sess.run(covs)
    sess.run(J)
    #print(i, J.eval())
    print(i, means.eval())
  
#covs = assemble_covs(covs_lower, num_components)

# Check the trained parameters with actual mean and covariance using numpy
print('Cost: {}, alpha={}, mean={}, cov={}'.format(J.eval(), alphas.eval(), means.eval(), covs.eval()))

print("Actual mean {0}".format(np.mean(dataset, 1)))
print("Actual cov {0}".format(np.cov(dataset)))

plot_clustered_data(dataset, means.eval(), covs.eval())
sess.close()
