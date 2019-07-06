from qgmm_utils import *
from draw_utils import *
from param_utils import *
import numpy as np
import tensorflow as tf
import pandas as pd
import math

def optimize():
  for _ in range(n_iter):
    _ = sess.run(training_op)
    sess.run(alphas)
    sess.run(means)
    sess.run(covs)
    sess.run(J)

# Load 'Old faithful' dataset
df = pd.read_csv('faithful.csv', sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

#obs = tf.constant([[1,2], [2,3]], dtype=tf.float32)

# Set the number of components is 2.
num_components = 2

m1 = get_initial_means()
m2 = get_initial_means()
print(m1, m2)

# Initialize means and covariances.
alphas = tf.Variable([0.37, 0.63], dtype=tf.float32, trainable=True)

means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
     dtype=tf.float32, trainable=True)

covs = tf.Variable([[[0.06916767, 0.0], [0.0, 20.69728207]], \
    [[0.06916767, 0.0], [0.0, 39.69728207]]], dtype=tf.float32, trainable=True)

# Calculate normalized gaussians
G = []
for i in range(num_components):
  G.append(unnormalized_gaussians(obs, means[i], covs[i]))
G = tf.convert_to_tensor(G, dtype=tf.float32)

P = quantum_gmm(obs, G, alphas, num_components)

#P = tf.div(P, tf.reduce_sum(P))
Q = get_Q(G, alphas, num_components)
Q = tf.stop_gradient(Q)
#Q = get_Q(G, alphas, num_components)

# lambda
ld = 0.01

# learning rate
lr = 0.01

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0

J = tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10))) + \
    ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + \
    2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * \
    tf.reduce_sum(G[0] * G[1])) - 1)#- tf.reduce_sum(P)) #1)
'''
J = tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10)))
'''
# Set optimizer to Adam with learning rate 0.01
optim = tf.train.AdamOptimizer(learning_rate=lr)
training_op = optim.minimize(-J)

# Build session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)

# Set the number of iterations is 2000.
n_iter = 100

best_J = -2e9
best_alphas = None
best_means = None
best_covs = None

plot_clustered_data(dataset, means.eval(), covs.eval(),\
    "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter), 0)

for i in range(500):
  optimize()    
  best_J = J.eval()
  best_alphas = alphas.eval()
  best_means = means.eval()
  best_covs = covs.eval()
  plot_clustered_data(dataset, best_means, best_covs, \
    "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter), i+1)
  
  #print(i, Q.eval())
  print(i, sess.run(tf.reduce_sum(P)))

# Check the trained parameters with actual mean and covariance using numpy
print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.\
    format(best_J, best_alphas, best_means, best_covs))

generate_video()

sess.close()
