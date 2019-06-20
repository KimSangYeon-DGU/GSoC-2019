import numpy as np

class QuantumGaussianDistribution:
  mean = []
  cov = []
  invCov = []
  covLower = []
  logDetCov = 0
  log2pi = 1.83787706640934533908193770912475883

  def __init__(self, mean, cov):
    print("Initialize QuantumGaussianDistribution")
    self.mean = np.asarray(mean)
    self.cov = np.asmatrix(cov)
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
    #print(observations.shape)
    observations = np.asarray(observations)
    k = observations.shape[1]
    diff = observations - self.mean
    
    v = np.dot(np.dot(diff, self.invCov), np.transpose(diff))
    #return -0.25 * k * self.log2pi - 0.25 * self.logDetCov - 0.25 * v
    return -0.5 * (k * self.log2pi + self.logDetCov + v)

  def Mean(self, mean):
    self.mean = mean
  
  def Covariance(self, cov):
    self.cov = cov

  def FactorCovariance(self):
    # To decompose matrix using Cholesky decomposition, apply a positive definite constraint. 
    self.ApplyPositiveDefinite(self.cov)

    covLower = np.linalg.cholesky(self.cov)

    invCovLower = np.linalg.inv(covLower)

    self.invCov = np.dot(np.transpose(invCovLower), invCovLower)
    _, self.logDetCov = np.linalg.slogdet(covLower)
    
    self.logDetCov *= 2

  def ApplyPositiveDefinite(self, cov):
    eigval, eigvec = np.linalg.eigh(cov)

    if (eigval[0] < 0.0) or ((eigval[eigval.shape[0] - 1] / eigval[0]) > 1e5) or (eigval[eigval.shape[0] - 1] < 1e-50):
      minEigval = max(eigval[eigval.shape[0] - 1] / 1e5, 1e-50)

      for i in range(eigval.shape[0]):
        eigval[i] = max(eigval[i], minEigval)
      
      self.cov = np.dot(np.dot(eigvec, np.diag(eigval)), np.transpose(eigvec))
      
    