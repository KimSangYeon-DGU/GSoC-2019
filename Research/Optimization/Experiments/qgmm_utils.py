import numpy as np
import tensorflow as tf

log2pi = 1.83787706640934533908193770912475883

def factor_covariance(covariance):
  cov_lower = tf.linalg.cholesky(covariance)
  inv_cov_lower = tf.linalg.inv(cov_lower)

  inv_cov = tf.transpose(inv_cov_lower) * inv_cov_lower
  _, log_det_cov = tf.linalg.slogdet(cov_lower)

  log_det_cov *= 2

  return inv_cov, log_det_cov

def log_probability(observations, mean, covariance, c):
  observations = tf.transpose(observations)
  
  k = tf.shape(observations)[1]
  
  inv_cov, log_det_cov = factor_covariance(covariance)
  
  diff = (observations - mean)
  return diff
  
  #v = tf.matmul(tf.matmul(diff, inv_cov), tf.transpose(diff))

  #return -c * (k * log2pi + log_det_cov + v)

# observations - N x N Matrix observations to used in calculating probability
# return - N unnormalized gaussian probabilities vector
def unnormalized_gaussians(observations, mean, covariance):
  obs = np.asmatrix(observations)
  obs_len = obs.shape[1]
  probabilities = []
  
  for i in range(obs_len):      
    probabilities.append(np.exp(log_probability(observations[:, i], mean, covariance, 0.25)))
  
  probabilities = np.asarray(probabilities)
  return probabilities

def get_cosine(G, alphas):
  return (1 - (alphas[0] ** 2) - (alphas[1] ** 2)) / (2 * alphas[0] * alphas[1] * np.sum(G[0] * G[1]))

def quantum_gmm(observations, alphas, means, covariances):
  G = []
  for i in range(len(alphas)):
    G.append(unnormalized_gaussians(observations, means[i], covariances[i]))
  G = np.asarray(G)

  prob_sum = 0
  print(get_cosine(G, alphas))
  for i in range(len(alphas)):
    prob_sum += (alphas[i] ** 2) * (G[i] ** 2) + (alphas[0] * alphas[1] * get_cosine(G, alphas) * G[0] * G[1])
  
  # Normalize the probability.
  prob_sum /= np.sum(prob_sum)

  return prob_sum

def constraint(observations, alphas, means, covariances):
  G = []
  for i in range(len(alphas)):
    G.append(unnormalized_gaussians(observations, means[i], covariances[i]))

  return (alphas[0] ** 2) + (alphas[1] ** 2) + 2 * alphas[0] * alphas[1] * get_cosine(G, alphas) * np.sum(G[0] * G[1])

# Calculate responsibilities.
def get_Q(G, alphas):
  o = (G[0] * G[1]) / np.sum(G[0] * G[1])
  alphao = (1 - (alphas[0] ** 2) - (alphas[1] ** 2)) * o

  Q = []
  common_divisor = (alphas[0] ** 2) * (G[0] ** 2) + (alphas[1] ** 2) * (G[1] ** 2) + alphao
  for i in range(len(alphas)):
    Q.append((alphas[i] ** 2) * (G[i] ** 2) + 0.5 * alphao)
  Q /= common_divisor

  return Q
