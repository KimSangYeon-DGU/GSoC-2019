import numpy as np
import tensorflow as tf

log2pi = 1.83787706640934533908193770912475883

def apply_positive_definite_constraint(covariance, num_components):
  eigval, eigvec = tf.linalg.eigh(covariance)
  
  eig_min = tf.reduce_min(eigval)
  eig_max = tf.reduce_max(eigval)
  
  def true():
    minEigval = tf.math.maximum(eig_max / 1e5, 1e-50)

    tmp_eigval = []
    for i in range(num_components):
      tmp_eigval.append(tf.math.maximum(eigval[i], minEigval))
    
    cov = tf.matmul(tf.matmul(eigvec, tf.diag(tmp_eigval)), tf.transpose(eigvec))
    return cov

  def false():
    return covariance

  cov = tf.cond(eig_min < 0.0, true, false)
  cov = tf.cond(eig_max / eig_min < 0.0, true, false)
  cov = tf.cond(eig_max  < 1e-50, true, false)    

  return cov

def factor_covariance(covariance):
  #cov = apply_positive_definite_constraint(covariance, 2)
  #cov_lower = tf.linalg.cholesky(cov)
  
  #inv_cov_lower = tf.linalg.inv(cov_lower)

  #inv_cov = tf.transpose(inv_cov_lower) * inv_cov_lower
  #_, log_det_cov = tf.linalg.slogdet(cov_lower)
  #log_det_cov *= 2
  
  inv_cov = tf.diag((1 / tf.diag_part(covariance)))
  log_det_cov = tf.reduce_sum(tf.math.log(tf.diag_part(covariance)))

  return inv_cov, log_det_cov

def log_probability(observations, mean, covariance, c):
  observations = tf.transpose(observations)
  k = tf.cast(tf.shape(observations)[1], tf.float32)

  inv_cov, log_det_cov = factor_covariance(covariance)
  
  diff = observations - mean
  v = tf.diag_part(tf.matmul(tf.matmul(diff, inv_cov), tf.transpose(diff)))

  return -c * (k * log2pi + log_det_cov + v)

# observations - N x N Matrix observations to used in calculating probability
# return - N unnormalized gaussian probabilities vector
def unnormalized_gaussians(observations, mean, covariance):
  #covariance = tf.matmul(tf.transpose(covariance), covariance)
  return tf.exp(log_probability(observations, mean, covariance, 0.25))

def get_cosine(G, alphas):
  return (1 - (alphas[0] ** 2) - (alphas[1] ** 2)) / (2 * alphas[0] * alphas[1] * tf.reduce_sum(G[0] * G[1]))

def quantum_gmm(observations, alphas, means, covariances, num_components):
  G = []
  
  for i in range(num_components):
    G.append(unnormalized_gaussians(observations, means[i], covariances[i]))
  G = tf.convert_to_tensor(G, dtype=tf.float32)

  prob_sum = 0
  
  for i in range(num_components):
    prob_sum += (alphas[i] ** 2) * (G[i] ** 2) + (alphas[0] * alphas[1] * get_cosine(G, alphas) * G[0] * G[1])
  
  return prob_sum

def constraint(observations, alphas, means, covariances, num_components):
  G = []
  for i in range(num_components):
    G.append(unnormalized_gaussians(observations, means[i], covariances[i]))
  G = tf.convert_to_tensor(G, dtype=tf.float32)

  return (alphas[0] ** 2) + (alphas[1] ** 2) + 2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * tf.reduce_sum(G[0] * G[1])

# Calculate responsibilities.
def get_Q(G, alphas, num_components):
  o = (G[0] * G[1]) / tf.reduce_sum(G[0] * G[1])
  alphao = (1 - (alphas[0] ** 2) - (alphas[1] ** 2)) * o

  Q = []
  common_divisor = (alphas[0] ** 2) * (G[0] ** 2) + (alphas[1] ** 2) * (G[1] ** 2) + alphao
  for i in range(num_components):
    Q.append((alphas[i] ** 2) * (G[i] ** 2) + 0.5 * alphao)
  Q /= common_divisor
  Q = tf.convert_to_tensor(Q, dtype=tf.float32)

  return Q

def assemble_covs(covs_lower, num_components):
  covs = []
  for i in range(num_components):
    covs.append(tf.matmul(tf.transpose(covs_lower[i]), covs_lower[i]))
  covs = tf.convert_to_tensor(covs, dtype=tf.float32)

  return covs