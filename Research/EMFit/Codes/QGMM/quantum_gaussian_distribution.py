import numpy as np

class QuantumGaussianDistribution:
  mean = []
  cov = []
  invCov = []
  covLower = []
  logDetCov = 0
  log2pi = 1.83787706640934533908193770912475883

  def __init__(self, mean, cov):
    self.mean = np.asarray(mean)
    self.cov = np.asmatrix(cov)

    # To decompose matrix using Cholesky decomposition, apply a positive definite constraint. 
    self.ApplyPositiveDefinite(self.cov)
    self.FactorCovariance()

  # Calculate probability
  def Probability(self, observations):
    obs = np.asmatrix(observations)
    obs_len = obs.shape[1]
    probabilities = []

    for i in range(obs_len):      
      tmp = self.LogProbability(observations[:, i])
      tmp = np.asarray(tmp[0])
      probabilities.append(np.exp(tmp[0][0]))
    
    probabilities = np.asarray(probabilities)
    return probabilities

  # Calculate log-probability
  def LogProbability(self, observations):
    observations = np.transpose(observations)
    
    observations = np.asarray(observations)
    
    k = len(observations)
    
    diff = observations - self.mean
    
    v = np.dot(np.dot(diff, self.invCov), np.transpose(diff))
    
    return -0.25 * (k * self.log2pi + self.logDetCov + v)

  def Mean(self, mean):
    self.mean = mean
  
  def Covariance(self, cov):
    self.cov = cov
    self.ApplyPositiveDefinite(self.cov)
    self.FactorCovariance()

  def FactorCovariance(self):
    self.covLower = np.linalg.cholesky(self.cov)
    #print(self.cov.shape)

    invCovLower = np.linalg.inv(self.covLower)

    self.invCov = np.dot(np.transpose(invCovLower), invCovLower)
    _, self.logDetCov = np.linalg.slogdet(self.covLower)
    
    self.logDetCov *= 2

  # Apply matrix to positive definite to use Cholesky decomposition.
  def ApplyPositiveDefinite(self, cov):
    eigval, eigvec = np.linalg.eigh(cov)

    if (eigval[0] < 0.0) or ((eigval[eigval.shape[0] - 1] / eigval[0]) > 1e5) or (eigval[eigval.shape[0] - 1] < 1e-50):
      minEigval = max(eigval[eigval.shape[0] - 1] / 1e5, 1e-50)

      for i in range(eigval.shape[0]):
        eigval[i] = max(eigval[i], minEigval)
      
      self.cov = np.dot(np.dot(eigvec, np.diag(eigval)), np.transpose(eigvec))
  
  # Sample observations with mean and covariance of the distribution.
  def Random(self):
    a = np.dot(self.covLower, np.random.normal(0, 1, (self.mean.shape[0], 1))) + np.reshape(self.mean, (self.mean.shape[0],1))
    return np.array(a).flatten()

  def Train(self, observations):
    mean = np.zeros((observations.shape[0], 1))
    cov = np.zeros((observations.shape[0], observations.shape[0]))
    
    for i in range(observations.shape[1]):
      mean += np.reshape(observations[:, i], (observations.shape[0], 1))

    mean /= observations.shape[1]

    for i in range(observations.shape[1]):
      diff = np.reshape(observations[:, i], (observations.shape[0], 1)) - mean
      cov += np.dot(diff, np.transpose(diff))

    cov /= (observations.shape[1] - 1)
    
    self.ApplyPositiveDefinite(cov)

    self.FactorCovariance()
