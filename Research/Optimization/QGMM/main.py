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
    sess.run(training)

# Load 'Old faithful' dataset
df = pd.read_csv('faithful.csv', sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
#dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
#print(np.max(dataset), np.min(dataset))
obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

# Set the number of components is 2.
gaussians = 2

#test_name = "t3_180"
test_name = "t1_1500_1"

m1, m2 = get_initial_means(dataset)

print(m1, m2)
# t1
m1 = [2.756031811312966, 76.62447648112042]
m2 = [2.9226572802266397, 88.3509418943818]

# t2
#m1 = [4.171021823127277, 83.66322004888708]
#m2 = [1.781079954983019, 95.411542531776]

# t3
#m1 = [4.616385494792178, 68.97139287485163]
#m2 = [4.73416217991247, 70.48443049223583]

# t4
#m1 = [3.5335808453329793, 60.79723193882826]
#m2 = [3.748786959785587, 46.017018024467745]

# t5
#m1 = [4.399318766072071, 63.982790484402784]
#m2 = [2.511548424664534, 90.2446329311453]

# Initialize means and covariances.
dimensionality = 2

alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True)

means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
    dtype=tf.float32, trainable=True)

phis = tf.Variable([45, -45], \
    dtype=tf.float32, trainable=True)

covs = tf.Variable([[[0.08, 0.1],
                     [0.1, 3.3]], \
                         
                    [[0.08, 0.1],
                     [0.1, 3.3]]], dtype=tf.float32, trainable=True)

# Calculate normalized gaussians
G = []
for i in range(gaussians):
  G.append(unnormalized_gaussians(obs, means[i], covs[i], gaussians))
G = tf.convert_to_tensor(G, dtype=tf.float32)
P = quantum_gmm(obs, G, alphas, gaussians, phis)

Q = get_Q(P, gaussians)
Q = tf.stop_gradient(Q)

# lambda
ld = 250

# learning rate
lr = 0.01

# Objective function :: Minimize (NLL + lambda * approximation constant)
# Approximation constant :: (Sum of P(p_{i}, k| alpha_{k}, theta_{k})) - 1 = 0
def loglikeihood(Q, P, gaussians):
    log_likelihood = 0
    for k in range(gaussians):
        log_likelihood += Q[k] * tf.math.log(tf.clip_by_value(P[k], 1e-10, 1e10))
    
    return tf.reduce_sum(log_likelihood)


def approx_constraint(G, alphas, phis):
    mix_sum = 0
    for k in range(gaussians):
        for l in range(gaussians):
            if k == l:
                continue
            mix_sum += alphas[k] * alphas[l] * G[k] * G[l] * get_cosine(phis[k] - phis[l])
    return tf.math.abs(tf.reduce_sum(alphas ** 2) + tf.reduce_sum(mix_sum) - 1)

# Objective function
J = -loglikeihood(Q, P, gaussians) + ld * approx_constraint(G, alphas, phis)

# Set optimizer to Adam with learning rate 0.01
optim = tf.train.AdamOptimizer(learning_rate=lr)
training = optim.minimize(J)

# Build session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)

# Set the number of iterations is 2000.
n_iter = 100

plot_clustered_data(dataset, means.eval(), covs.eval(), test_name, 0)

# For graph
max_iteration = 500

xs = []
ys = []
cs = []
as1 = []
as2 = []
Phis = []
G1s = []
G2s = []
Ps = []
Js = []

NLL = -loglikeihood(Q, P, gaussians).eval()
C = approx_constraint(G, alphas, phis).eval()
cur_alphas = alphas.eval()

# Save initial values
xs.append(0)
ys.append(NLL)
cs.append(C)
as1.append(cur_alphas[0])
as2.append(cur_alphas[1])
cur_phi = phis.eval()
Phis.append(cur_phi)
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
  print(i, test_name, tf.reduce_sum(P[0] + P[1]).eval())
  print(phis.eval(), cur_J)
  optimize()
  cur_J = J.eval()
  cur_alphas = alphas.eval()
  cur_means = means.eval()
  cur_covs = covs.eval()

  plot_clustered_data(dataset, cur_means, cur_covs, test_name, i)
  
  # Save values for graphs
  xs.append(i * n_iter)
  ys.append(-loglikeihood(Q, P, gaussians).eval())
  cs.append(approx_constraint(G, alphas, phis).eval())
  as1.append(cur_alphas[0])
  as2.append(cur_alphas[1])
  cur_phi = get_cosine(phis).eval()
  Phis.append(cur_phi)
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
phi_file_name = '{0}_phi'.format(test_name)
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

draw_graph(x=xs, y=ys, x_label='Iteration', y_label='NLL', 
    file_name=nll_file_name+'.png', test_name=test_name)

# Constraint
with open(csv_path+'/'+constraint_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, cs))
draw_graph(x=xs, y=cs, x_label='Iteration', y_label='Constraint', 
    file_name=constraint_file_name+'.png', test_name=test_name)

# Phi
with open(csv_path+'/'+phi_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, Phis))

draw_graph(x=xs, y=Phis, x_label='Iteration', y_label="Phi", \
    file_name=phi_file_name+'.png', test_name=test_name)

# Alpha
with open(csv_path+'/'+alpha_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, as1, as2))

draw_alphas_graph(x=xs, a1=as1, a2=as2, x_label='Iteration', y_label="Alphas",\
    file_name=alpha_file_name, test_name=test_name)

# Unnormalized Gaussians
with open(csv_path+'/'+unnorm_gauss_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, G1s, G2s))
draw_gaussian(x=xs, g1=G1s, g2=G2s, x_label='Iteration', \
    y_label='Unnormalized Gaussians', g1_label='G1', g2_label='G2', \
    file_name=unnorm_gauss_file_name+'.png', test_name=test_name)

# Probability
with open(csv_path+'/'+probs_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, Ps))

draw_graph(x=xs, y=Ps, x_label='Iteration', y_label='Sum of probability', 
    file_name=probs_file_name+'.png', test_name=test_name)

# Objective function
with open(csv_path+'/'+obj_file_name, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(zip(xs, Js))

draw_graph(x=xs, y=Js, x_label='Iteration', y_label='Objective function', 
    file_name=obj_file_name+'.png', test_name=test_name)

# Generate a video
generate_video(test_name)

sess.close()
