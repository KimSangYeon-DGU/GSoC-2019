from gaussian_distribution import GaussianDistribution
from gmm import GMM
import numpy as np

################################
## Gaussian Distribution Test ##
################################

def QuantumGaussianDistributionProbabilityTest():
  # Set initial parameters
  mean1 = np.array( [5, 6, 3, 3, 2] )

  cov1 = np.matrix( [[6, 1, 1, 1, 2],
                     [1, 7, 1, 0, 0],
                     [1, 1, 4, 1, 1],
                     [1, 0, 1, 7, 0],
                     [2, 0, 1, 0, 6]] )

  # Set observations to train
  observations = np.matrix( [[0],
                             [1],
                             [2],
                             [3],
                             [4]] )

  print(observations[:, 0])
  # Create distributions
  d1 = GaussianDistribution(mean1, cov1)
  mean1 *= -1
  print(d1.Probability(observations)) 

def GaussianDistributionRandomTest():
  mean = np.array([1.0, 2.25])
  cov = np.matrix([[0.85, 0.60],
                   [0.60, 1.45]])

  d = GaussianDistribution(mean, cov)

  obs = np.zeros((2, 5000))

  for i in range(5000):
    obs[:,i] = d.Random()

  np_mean = np.mean(obs, 1)
  
  np_cov = np.cov(obs)

  print(np_mean)
  print(np_cov)

def GaussianDistributionTrainTest():
  mean = np.array([1, 3, 0, 2.5])
  cov = np.matrix([[3.0, 0.0, 1.0, 4.0],
                   [0.0, 2.4, 0.5, 0.1],
                   [1.0, 0.5, 6.3, 0.0],
                   [4.0, 0.1, 0.0, 9.1]])

  dummy_mean = np.array([0, 0, 0, 0])
  dummy_cov = np.matrix([[0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])

  obs = np.zeros((4, 10000))

  d1 = GaussianDistribution(mean, cov)

  for i in range(10000):
    obs[:,i] = d1.Random()

  d2 = GaussianDistribution(mean, cov)

  d2.Train(obs)

  print(d2.mean)
  print(d2.cov)

##################################
## Gaussian Mixture Models Test ##
##################################

if __name__ == "__main__":
  #GaussianDistributionRandomTest()
  #GaussianDistributionTrainTest()

  mat = np.zeros((5, 5))
  v = np.array( [1,2,3,4,5] )

  mat[2] = v

  print(mat)

  