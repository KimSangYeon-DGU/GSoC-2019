from qgmm_utils import *
import numpy as np
import tensorflow as tf

num_components = 2

alphas = tf.Variable([0.3, 0.7], dtype=tf.float32)
means = tf.Variable([[1.0, 1.0], [0.0, 0.0]], dtype=tf.float32)
covs = tf.Variable([[[1.0, 0.0], [0.0, 1.0]],[[2.0, 0.0],[0.0, 2.0]]], dtype=tf.float32)
covs_lower = tf.linalg.cholesky(covs)

obs = np.array([[1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6]])

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

'''
for i in range(alphas.shape[0]):
  cov = tf.transpose(covs_lower[i]) * covs_lower[i]
  print(sess.run(cov))
'''
cov = tf.transpose(covs_lower[0]) * covs_lower[0]
print(sess.run(log_probability(obs, means[0], cov, 0.25)))
#print(unnormalized_gaussians(obs, means[0], cov))

#G = unnormalized_gaussians(obs, means[0], covs[0])
#print(sess.run(tf_alphas[0]))
'''
G = []
for alpha, mean, cov_lower in zip(alphas, means, covs_lower):
  cov = np.dot(cov_lower.T, cov_lower)
  G.append(unnormalized_gaussians(obs, mean, cov))
'''
'''
Q = get_Q(G, alphas)
G = np.asarray(G)
Q = np.asarray(Q)
ld = 0.1

G = tf.convert_to_tensor(G, dtype=tf.float32)
Q = tf.convert_to_tensor(Q, dtype=tf.float32)

J = tf.reduce_sum(Q[0] * tf.log(G[0]) + Q[1] * tf.log(G[1]))# + ld * (((tf_alphas[0] ** 2) + (tf_alphas[1] ** 2) + 2 * tf_alphas[0] * tf_alphas[1] * get_cosine(G, tf_alphas) * np.sum(G[0] * G[1])) - 1)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

ALPHA, MEAN, COV, C = [], [], [], []
sess.run(init)
n_iter = 2000

optim = tf.train.RMSPropOptimizer(learning_rate=0.01)
training_op = optim.minimize(-J)


for i in range(n_iter):
    _ = sess.run(training_op)
    ALPHA.append(sess.run(tf_alphas))
    MEAN.append(sess.run(tf_means))
    COV.append(sess.run(tf_covs_lower))
    C.append(sess.run(J))

# Get the final values    
print('after {} iterations:'.format(n_iter))
print('Cost: {} at Alpha={}, Mean={}, Cov={}'.format(J.eval(), tf_alphas.eval(), tf_means.eval(), tf_covs_lower.eval()))

sess.close()
'''