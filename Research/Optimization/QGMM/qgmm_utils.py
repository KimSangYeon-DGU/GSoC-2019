import numpy as np
import tensorflow as tf

log2pi = 1.83787706640934533908193770912475883

def apply_positive_definite_constraint(covariance, dimensionality):
  try:
    eigval, eigvec = tf.linalg.eigh(covariance)
  except Exception:
    print("Eigen decomposition error:{0}".format(covariance))

  eig_min = tf.reduce_min(eigval)
  eig_max = tf.reduce_max(eigval)
  check_val = 0

  def T():
    return 1
  def F():
    return 0

  check_val += tf.cond(eig_min < 0.0, T, F)
  check_val += tf.cond(eig_max / eig_min < 0.0, T, F)
  check_val += tf.cond(eig_max  < 1e-50, T, F)    

  def apply():
    minEigval = tf.math.maximum(eig_max / 1e5, 1e-50)

    tmp_eigval = []
    for i in range(dimensionality):
      tmp_eigval.append(tf.math.maximum(eigval[i], minEigval))
    
    cov = tf.matmul(tf.matmul(eigvec, tf.diag(tmp_eigval)), \
        tf.transpose(eigvec))
    return cov
  
  def disapply():
    return covariance
  
  return tf.cond(0 < check_val, apply, disapply)

def factor_covariance(covariance, dimensionality):
  covariance = apply_positive_definite_constraint(covariance, dimensionality)
  cov_lower = tf.linalg.cholesky(covariance)
  inv_cov_lower = tf.linalg.inv(cov_lower)
  
  inv_cov = tf.matmul(tf.transpose(inv_cov_lower), inv_cov_lower)
  _, log_det_cov = tf.linalg.slogdet(cov_lower)
  log_det_cov *= 2
  
  '''
  # For using when we use diagonal constraint.
  inv_cov = tf.diag((1 / tf.diag_part(covariance)))
  log_det_cov = tf.reduce_sum(tf.math.log(tf.diag_part(covariance)))
  '''

  return inv_cov, log_det_cov

def compose_covariance(covariance):
  lower_cov = tf.matrix_band_part(covariance, -1, 0)
  return tf.matmul(lower_cov, tf.transpose(lower_cov))

def log_probability(observations, mean, covariance, c, dimensionality):
  observations = tf.transpose(observations)
  k = tf.cast(tf.shape(observations)[1], tf.float32)

  inv_cov, log_det_cov = factor_covariance(covariance, dimensionality)
  
  diff = observations - mean
  v = tf.diag_part(tf.matmul(tf.matmul(diff, inv_cov), tf.transpose(diff)))

  return -c * (k * log2pi + log_det_cov + v)

# observations - N x N Matrix observations to used in calculating probability
# return - N unnormalized gaussian probabilities vector
def unnormalized_gaussians(observations, mean, covariance, dimensionality):
  #covariance = compose_covariance(covariance)
  return tf.exp(log_probability(observations, mean, covariance, \
      0.25, dimensionality))

def get_cosine(phis, k, l):
  pi_on_180 = 0.017453292519943295
  phi = phis[k] - phis[l]
  phi = phi * pi_on_180
  return tf.cos(phi) # deg2rad

def quantum_gmm(observations, G, alphas, gaussians, phis):
  P = []
  
  for k in range(gaussians):
    probs = alphas[k] * G[k]
    probs_sum = 0
    for l in range(gaussians):
      probs_sum += alphas[l] * get_cosine(phis, k, l)* G[l]
    P.append(probs * probs_sum)
  return P

# Calculate responsibilities.
def get_Q(P, gaussians):
  probs_sum = 0
  for i in range(gaussians):
    probs_sum += P[i]

  return tf.div(P, probs_sum)
