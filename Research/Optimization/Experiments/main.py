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
    sess.run(means)
    sess.run(covs)
    sess.run(alphas)
    sess.run(J)

# Load 'Old faithful' dataset
df = pd.read_csv('faithful.csv', sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

# Set the number of components is 2.
num_components = 2

m1 = get_initial_means()
m2 = get_initial_means()
print(m1, m2)

# Cases to solve later for stability of the training
# 1) Diverged case
#m1 = [3.478331200843432, 75.77990594394055]
#m2 = [1.402800973631848, 85.03263955392423]

# 2) Variance vanishing case
#m1 = [4.14836138113223, 61.47837516815865]
#m2 = [3.8196297714372522, 75.9772116104858]

# 3) One large cluster case
#m1 = [1.127841276077628, 94.31239215734522]
#m2 = [4.055697916182589, 65.35135198155628]

# 4) Input is not invertible
m1 = [4.115197647253183, 30.519398980416106]
m2 = [2.2308616562224004, 59.32051981012918]

# Initialize means and covariances.
alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True)

means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
     dtype=tf.float32, trainable=True)

covs = tf.Variable([[[0.1, 0.0], [0.0, 5.0]], \
    [[0.1, 0.0], [0.0, 5.0]]], dtype=tf.float32, trainable=True)

# Calculate normalized gaussians
G = []
for i in range(num_components):
  G.append(unnormalized_gaussians(obs, means[i], covs[i], num_components))
G = tf.convert_to_tensor(G, dtype=tf.float32)
P = quantum_gmm(obs, G, alphas, num_components)

Q = get_Q(G, alphas, num_components)
Q = tf.stop_gradient(Q)

# lambda
ld = 0.01

# learning rate
lr = 0.001

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0
'''
J = -1 * tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10))) + \
    ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + \
    2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * \
    tf.reduce_sum(G[0] * G[1])) - 1)#- tf.reduce_sum(P)) #1)
'''
'''
J = -1 * tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10)))
'''
'''
J = -1 * tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10))) + \
    ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + 2 * alphas[0] * \
    tf.reduce_sum(G[0] ** 2) * alphas[1] * tf.reduce_sum(G[1] ** 2) * \
    get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])) - 1)
'''

a = alphas[0] * tf.reduce_sum(G[0] ** 2) - alphas[1] * tf.reduce_sum(G[1] ** 2)
b = alphas[0] * tf.reduce_sum(G[0] ** 2) + alphas[1] * tf.reduce_sum(G[1] ** 2)
alpha_max_const = tf.math.maximum(a, b) - 1
alpha_min_const = tf.math.minimum(a, b) - 1

J = -1 * tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10))) + \
    ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + 2 * alphas[0] * \
    tf.reduce_sum(G[0] ** 2) * alphas[1] * tf.reduce_sum(G[1] ** 2) * \
    get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])) - 1)

# Set optimizer to Adam with learning rate 0.01
optim = tf.train.AdamOptimizer(learning_rate=lr)
#optim = tf.train.RMSPropOptimizer(learning_rate=lr)
training_op = optim.minimize(J)

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

# For graph
xs = []
ys = []

for i in range(500):
  optimize()
  best_J = J.eval()
  best_alphas = alphas.eval()
  best_means = means.eval()
  best_covs = covs.eval()
  plot_clustered_data(dataset, best_means, best_covs, \
    "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter), i+1)
  
  xs.append(i)
  ys.append(best_J)
  print("{0} G1**2: {1}, G2**2: {2}".format(i, tf.reduce_sum(G[0]**2).eval(), tf.reduce_sum(G[1]**2).eval() ) )
  #print("added_loss: {0}".format(added_loss.eval()))
  '''
  print("{0}, NLL:{1}, Costraint:{2}".format(i, (tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) + \
    Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10)))).eval(),  ld * (((alphas[0] ** 2) + (alphas[1] ** 2) + \
    2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * \
    tf.reduce_sum(G[0] * G[1])) - 1).eval() ) )
  '''

# Check the trained parameters with actual mean and covariance using numpy
print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.\
    format(best_J, best_alphas, best_means, best_covs))

draw_graph(x=xs, y=ys, x_label='iteration', y_label='NLL', file_name='NLL.png')

generate_video()

sess.close()
