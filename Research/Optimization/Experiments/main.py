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

#m1 = get_initial_means()
#m2 = get_initial_means()
m1, m2 = get_initial_means_from_dataset(dataset)
#m1 = [2.3, 75]
#m2 = [3.5, 85]
print(m1, m2)

# Easy case
#m1 = [2.527462637671865, 65.03104989882695]
#m2 = [4.541881277584392, 75.60486840558593]

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
#m1 = [4.115197647253183, 30.519398980416106]
#m2 = [2.2308616562224004, 59.32051981012918]

# Initialize means and covariances.
alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True)

means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
     dtype=tf.float32, trainable=True)


covs = tf.Variable([[[0.3, 0.0], [0.1, 5.3]], \
    [[0.3, 0.0], [0.1, 5.3]]], dtype=tf.float32, trainable=True)

# Calculate normalized gaussians
G = []
for i in range(num_components):
  G.append(unnormalized_gaussians(obs, means[i], covs[i], num_components))
G = tf.convert_to_tensor(G, dtype=tf.float32)
P = quantum_gmm(obs, G, alphas, num_components)

Q = get_Q(G, alphas, num_components)
Q = tf.stop_gradient(Q)

# lambda
ld = 1

# learning rate
lr = 0.001

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0
def loglikeihood(Q, P):
  return tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) \
      + Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10)))

'''
def approx_constraint(G, alphas):
  return tf.math.abs( ((alphas[0] ** 2) + (alphas[1] ** 2)) - 1)

def approx_constraint(G, alphas):
  return ((alphas[0] ** 2) + (alphas[1] ** 2) \
    + 2 * alphas[0] * alphas[1] \
    * get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])) - 1
'''

def approx_constraint(G, alphas):
  return tf.math.abs( ((alphas[0] ** 2) * tf.reduce_sum(G[0] ** 2) + (alphas[1] ** 2) \
    * tf.reduce_sum(G[1] ** 2) + 2 * alphas[0] * alphas[1] \
    * get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])) - 1)


J = -loglikeihood(Q, P) + ld * approx_constraint(G, alphas)
#J = -loglikeihood(Q, P) + ld * approx_constraint(G, alphas) + 10 * (tf.reduce_sum(G[0]) + tf.reduce_sum(G[1]))


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
max_iteration = 500
xs = []
ys = []
cs = []
as1 = []
as2 = []

NLL = -loglikeihood(Q, P).eval()
C = approx_constraint(G, alphas).eval()
cur_alphas = alphas.eval()

# Save initial values
xs.append(0)
ys.append(NLL)
cs.append(C)
as1.append(cur_alphas[0])
as2.append(cur_alphas[1])

tot = 1e-4
sess.run(J)
cur_J = J.eval()
pre_J = cur_J
# Train QGMM
for i in range(1, max_iteration):
  optimize()
  print(i)
  cur_J = J.eval()
  cur_alphas = alphas.eval()
  cur_means = means.eval()
  cur_covs = covs.eval()
  NLL = -loglikeihood(Q, P).eval()
  C = approx_constraint(G, alphas).eval()

  plot_clustered_data(dataset, cur_means, cur_covs, \
    "QGMM_last_{0}_{1}_{2}.png".format(ld, lr, n_iter), i+1)
  
  # Save values for graphs
  xs.append(i * n_iter)
  ys.append(NLL)
  cs.append(C)
  as1.append(cur_alphas[0])
  as2.append(cur_alphas[1])
  print("{0} G1**2: {1}, G2**2: {2}, alphas: {3}, mean: {6}, cov: {7}\nJ: {4}, C: {5}, lambda: {8}".format(i, tf.reduce_sum(G[0]**2).eval(), tf.reduce_sum(G[1]**2).eval(), cur_alphas, cur_J, C, cur_means, cur_covs, ld) )
  print(get_cosine(G, alphas).eval())
  print(tf.reduce_sum(P).eval())
  if abs(pre_J - cur_J) < tot:
    break
  pre_J = cur_J

# Check the trained parameters with actual mean and covariance using numpy
print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.\
    format(cur_J, cur_alphas, cur_means, cur_covs))

draw_graph(x=xs, y=ys, x_label='iteration', y_label='NLL with constraint', file_name='nll.png')
draw_graph(x=xs, y=cs, x_label='iteration', y_label='Constraint', file_name='constraint.png')
draw_alphas_graph(x=xs, a1=as1, a2=as2, x_label='iteration', y_label="alphas", file_name="alphas.png")
#generate_video()
generate_video2()

sess.close()
