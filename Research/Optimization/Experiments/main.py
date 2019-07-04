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

#obs = tf.constant([[1,2], [2,3]], dtype=tf.float32)

# Set the number of components is 2.
num_components = 2

#alphas = tf.Variable(tf.sqrt([1, 1.5]), dtype=tf.float32, trainable=True)
alphas = tf.Variable([0.6, 0.4], dtype=tf.float32, trainable=True)
#alphas = tf.Variable([1.0, 1.0], dtype=tf.float32)
# Initialize means and covariances.
#means = tf.Variable([[2.0638845, 82.47851638], [5.48966197, 50.96811517]], \
#     dtype=tf.float32, trainable=True)

means = tf.Variable([[2.0638845, 82.47851638], [5.48966197, 50.96811517]], \
     dtype=tf.float32, trainable=True)
covs = tf.Variable([[[0.06916767, 0.4], [0.4, 33.69728207]], \
    [[0.16996844, 0.3],[0.3, 36.04621132]]], dtype=tf.float32, trainable=True)
#covs = tf.Variable([[[0.06916767, 0.0], [0.0, 10.69728207]], \
#    [[0.046996844, 0.3],[0.3, 36.04621132]]], dtype=tf.float32, trainable=True)

# Calculate normalized gaussians
G = []
for i in range(num_components):
  G.append(unnormalized_gaussians(obs, means[i], covs[i]))

G = tf.convert_to_tensor(G, dtype=tf.float32)

#G = tf.div(G, tf.reduce_sum(G))
Q = get_Q(G, alphas, num_components)
Q = tf.stop_gradient(Q)
#Q = get_Q(G, alphas, num_components)

# lambda
#ld = 0.01
#ld = 1 / tf.reduce_sum(G)
ld = 0.0001

# learning rate
lr = 0.001

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0

J = tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(G[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(G[1], 1e-10, 1e10))) + \
    ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + \
    2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * \
    tf.reduce_sum(G[0] * G[1])) - tf.reduce_sum(G)) #1)
'''
J = tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(G[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(G[1], 1e-10, 1e10)))
'''
# Set optimizer to Adam with learning rate 0.1
optim = tf.train.AdamOptimizer(learning_rate=lr)
training_op = optim.minimize(-J)

# Build session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(tf.reduce_sum(G)))
plot_clustered_data(dataset, means.eval(), covs.eval(), "QGMM_first.png")

# Set the number of iterations is 2000.
n_iter = 100

best_J = -2e9
best_alphas = None
best_means = None
best_covs = None


def optimize():
  for _ in range(n_iter):
    _ = sess.run(training_op)
    sess.run(alphas)
    sess.run(means)
    sess.run(covs)
    #sess.run(J)

for i in range(1000):
  optimize()  
  sess.run(J)
  #print(J.eval())
  #print(i, alphas.eval() ** 2)
  
  best_J = J.eval()
  best_alphas = alphas.eval()
  best_means = means.eval()
  best_covs = covs.eval()
  plot_clustered_data(dataset, best_means, best_covs, \
    "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter))
  
  #print(i, Q.eval())
  print(sess.run(tf.reduce_sum(G)))
  # Save the parameters.
  #if math.isnan(J.eval()) != True and best_j < J.eval():
    #best_j = J.eval()
    #best_alphas = alphas.eval()
    #best_means = means.eval()
    #best_covs = covs.eval()
print(sess.run(tf.reduce_sum(G)))

# Check the trained parameters with actual mean and covariance using numpy
print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.\
    format(best_J, best_alphas, best_means, best_covs))

#plot_clustered_data(dataset, best_means, best_covs, \
#    "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter))

sess.close()
