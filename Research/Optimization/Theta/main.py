from qgmm_utils import *
from draw_utils import *
from param_utils import *
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import csv

def optimize():
  for _ in range(n_iter):
    _ = sess.run(training_op)
    sess.run(means)
    sess.run(covs)
    sess.run(alphas)
    sess.run(phi)
    sess.run(J)

# Load 'Old faithful' dataset
df = pd.read_csv('faithful.csv', sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

# Set the number of components is 2.
num_components = 2

test_name = "t1"

m1, m2 = get_initial_means(dataset)

print(m1, m2)

# t1 150
m1 = [4.635260552571155, 70.199885051446]
m2 = [4.533706042588, 87.74118169887785]

# t2, t3
#m1 = [1.6370093246808495, 58.1976610188544]
#m2 = [2.1763603108042435, 82.24250064700527]

# Initialize means and covariances.
alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True)
#alphas = tf.Variable(np.sqrt([0.5, 0.5]), dtype=tf.float32, trainable=True)

means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
    dtype=tf.float32, trainable=True)

phi = tf.Variable(np.deg2rad(0), \
    dtype=tf.float32, trainable=True)

#covs = tf.Variable([[[0.1, 0.01], [0.01, 6.3]], \
#    [[0.1, 0.01], [0.01, 6.3]]], dtype=tf.float32, trainable=True)

covs = tf.Variable([[[0.08, 0.1], [0.1, 3.3]], \
    [[0.08, 0.1], [0.1, 3.3]]], dtype=tf.float32, trainable=True)


# Calculate normalized gaussians
G = []
for i in range(num_components):
  G.append(unnormalized_gaussians(obs, means[i], covs[i], num_components))
G = tf.convert_to_tensor(G, dtype=tf.float32)
P = quantum_gmm(obs, G, alphas, num_components, phi)

Q = get_Q(G, alphas, num_components)
Q = tf.stop_gradient(Q)

# lambda
#ld = 53
#ld = 53 # 70 t3
#ld = 139
#ld = 84
#ld = 90 t1
ld = 217

# learning rate
lr = 0.01

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P) - 1 = 0
def loglikeihood(Q, P):
  return tf.reduce_sum(Q[0] * tf.math.log(tf.clip_by_value(P[0], 1e-10, 1e10)) \
      + Q[1] * tf.math.log(tf.clip_by_value(P[1], 1e-10, 1e10)))

'''
def approx_constraint(G, alphas, phi):
  return tf.math.abs( ((alphas[0] ** 2) * tf.reduce_sum(G[0] ** 2) \
    + (alphas[1] ** 2) * tf.reduce_sum(G[1] ** 2) + 2 * alphas[0] * alphas[1] \
    * get_cosine(phi) * tf.reduce_sum(G[0] * G[1])) - 1)
'''
def approx_constraint(G, alphas, phi):
  return tf.math.abs( ((alphas[0] ** 2) \
    + (alphas[1] ** 2) + 2 * alphas[0] * alphas[1] \
    * get_cosine(phi) * tf.reduce_sum(G[0] * G[1])) - 1)

J = -loglikeihood(Q, P) + ld * approx_constraint(G, alphas, phi)

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


plot_clustered_data(dataset, means.eval(), covs.eval(), test_name, 0)

# For graph
max_iteration = 350

xs = []
ys = []
cs = []
as1 = []
as2 = []
Cosines = []
G1s = []
G2s = []
Ps = []
Js = []

NLL = -loglikeihood(Q, P).eval()
C = approx_constraint(G, alphas, phi).eval()
cur_alphas = alphas.eval()

# Save initial values
xs.append(0)
ys.append(NLL)
cs.append(C)
as1.append(cur_alphas[0])
as2.append(cur_alphas[1])
cur_cosine = get_cosine(phi).eval()
Cosines.append(cur_cosine)
G1s.append(tf.reduce_sum(G[0]).eval())
G2s.append(tf.reduce_sum(G[1]).eval())
Ps.append(tf.reduce_sum(P[0] + P[1]).eval())

tot = 1e-3
sess.run(J)
cur_J = J.eval()
pre_J = cur_J

Js.append(cur_J)

# Train QGMM
for i in range(1, max_iteration):
  print(i, test_name, tf.reduce_sum(P[0] + P[1]).eval(), cur_cosine)
  optimize()
  cur_J = J.eval()
  cur_alphas = alphas.eval()
  cur_means = means.eval()
  cur_covs = covs.eval()

  plot_clustered_data(dataset, cur_means, cur_covs, test_name, i)
  
  # Save values for graphs
  xs.append(i * n_iter)
  ys.append(-loglikeihood(Q, P).eval())
  cs.append(approx_constraint(G, alphas, phi).eval())
  as1.append(cur_alphas[0])
  as2.append(cur_alphas[1])
  cur_cosine = get_cosine(phi).eval()
  Cosines.append(cur_cosine)
  #Cosines.append(tf.reduce_sum(get_cosine(G, alphas)).eval())
  G1s.append(tf.reduce_sum(G[0]).eval())
  G2s.append(tf.reduce_sum(G[1]).eval())
  Js.append(cur_J)

  Ps.append(tf.reduce_sum(P[0] + P[1]).eval())

  if abs(pre_J - cur_J) < tot:
    break
  pre_J = cur_J

# Check the trained parameters with actual mean and covariance using numpy
print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.\
    format(cur_J, cur_alphas, cur_means, cur_covs))

# Set file names
nll_file_name = '{0}_nll'.format(test_name)
constraint_file_name = '{0}_constraint'.format(test_name)
cos_file_name = '{0}_cos'.format(test_name)
unnorm_gauss_file_name = '{0}_unnorm_gauss'.format(test_name)
probs_file_name = '{0}_probs'.format(test_name)
alpha_file_name = '{0}_alpha'.format(test_name)
obj_file_name = '{0}_obj'.format(test_name)
csv_path = './csvs/{0}'.format(test_name)

# Save data to csv format and a graph
# NLL
with open(csv_path+'/'+nll_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, ys))

draw_graph(x=xs, y=ys, x_label='iteration', y_label='NLL', 
    file_name=nll_file_name+'.png', test_name=test_name)

# Constraint
with open(csv_path+'/'+constraint_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, cs))
draw_graph(x=xs, y=cs, x_label='iteration', y_label='Constraint', 
    file_name=constraint_file_name+'.png', test_name=test_name)

# Cosine
with open(csv_path+'/'+cos_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, Cosines))

draw_graph(x=xs, y=Cosines, x_label='iteration', y_label='cos(phi)', 
    file_name=cos_file_name+'.png', test_name=test_name)

# Alpha
with open(csv_path+'/'+alpha_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, as1, as2))

draw_alphas_graph(x=xs, a1=as1, a2=as2, x_label='iteration', y_label="alphas",\
    file_name=alpha_file_name, test_name=test_name)

# Unnormalized Gaussians
with open(csv_path+'/'+unnorm_gauss_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, G1s, G2s))
draw_gaussian(x=xs, g1=G1s, g2=G2s, x_label='iteration', \
    y_label='Unnormalized Gaussians', g1_label='G1', g2_label='G2', \
    file_name=unnorm_gauss_file_name+'.png', test_name=test_name)

# Probability
with open(csv_path+'/'+probs_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, Ps))

draw_graph(x=xs, y=Ps, x_label='iteration', y_label='probability', 
    file_name=probs_file_name+'.png', test_name=test_name)

# Objective function
with open(csv_path+'/'+obj_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, Js))

draw_graph(x=xs, y=Js, x_label='iteration', y_label='objective function', 
    file_name=obj_file_name+'.png', test_name=test_name)

# Generate a video
generate_video(test_name)

sess.close()
