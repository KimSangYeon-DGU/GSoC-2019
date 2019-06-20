from quantum_gaussian_distribution import QuantumGaussianDistribution
from quantum_gmm import QuantumGMM
import numpy as np

def ProbabilityTest():
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
  d1 = QuantumGaussianDistribution(mean1, cov1)
  mean1 *= -1
  print(d1.Probability(observations)) 

def RandomTest():
  mean = np.array([1.0, 2.25])
  cov = np.matrix([[0.85, 0.60],
                   [0.60, 1.45]])

  d = QuantumGaussianDistribution(mean, cov)

  obs = np.zeros((2, 5000))

  for i in range(5000):
    obs[:,i] = d.Random()

  np_mean = np.mean(obs, 1)
  
  np_cov = np.cov(obs)

  print(np_mean)
  print(np_cov)

if __name__ == "__main__":
  RandomTest()
  