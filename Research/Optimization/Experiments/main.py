from qgmm_utils import *
from draw_utils import *
import numpy as np
import tensorflow as tf
import pandas as pd
import math

# Load 'Old faithful' dataset
df = pd.read_csv('faithful.csv', sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

# Set the number of components is 2.
num_components = 2

alphas = tf.Variable([1.0, 1.0], dtype=tf.float32)

# Initialize means and covariances.
means = tf.Variable([[1.0638845, 52.47851638], [5.48966197, 79.96811517]], dtype=tf.float32, trainable=True)
covs = tf.Variable([[[0.06916767, 0.4], [0.4, 33.69728207]],[[0.16996844, 0.3],[0.3, 36.04621132]]], dtype=tf.float32, trainable=True)

# Calculate normalized gaussians
G = []
for i in range(num_components):
  G.append(unnormalized_gaussians(obs, means[i], covs[i]))

G = tf.convert_to_tensor(G, dtype=tf.float32)
G = tf.div(G, tf.reduce_sum(G))
Q = get_Q(G, alphas, num_components)

# lambda
ld = 0.1

# learning rate
lr = 0.01

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0
J = tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(G[0], 1e-10, 1e10)) + Q[1] * tf.math.log(tf.clip_by_value(G[1], 1e-10, 1e10))) + ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + 2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])) - 1)

# Set optimizer to Adam with learning rate 0.1
optim = tf.train.AdamOptimizer(learning_rate=lr)
training_op = optim.minimize(-J)

# Build session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)

plot_clustered_data(dataset, means.eval(), covs.eval(), "QGMM_first.png")

# Set the number of iterations is 2000.
n_iter = 120000

best_j = -2e9
best_alphas = None
best_means = None
best_covs = None
for i in range(n_iter):
    _ = sess.run(training_op)
    sess.run(alphas)
    sess.run(means)
    sess.run(covs)
    sess.run(J)
    print(i, means.eval())
    
    # Save the parameters.
    #if math.isnan(J.eval()) != True and best_j < J.eval():
      #best_j = J.eval()
      #best_alphas = alphas.eval()
      #best_means = means.eval()
      #best_covs = covs.eval()

best_j = J.eval()
best_alphas = alphas.eval()
best_means = means.eval()
best_covs = covs.eval()

# Check the trained parameters with actual mean and covariance using numpy
print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.format(best_j, best_alphas, best_means, best_covs))

plot_clustered_data(dataset, best_means, best_covs, "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter))

sess.close()
