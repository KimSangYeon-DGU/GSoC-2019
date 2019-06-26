from quantum_gaussian_distribution import QuantumGaussianDistribution
from quantum_gmm import QuantumGMM
from gaussian_distribution import GaussianDistribution
from gmm import GMM
import numpy as np

if __name__ == "__main__":

  # Set initial parameters
  mean1 = np.array( [5, 6, 3, 3, 2] )

  cov1 = np.matrix( [[6, 1, 1, 1, 2],
                     [1, 7, 1, 0, 0],
                     [1, 1, 4, 1, 1],
                     [1, 0, 1, 7, 0],
                     [2, 0, 1, 0, 6]] )

  mean2 = np.array( [6, 6, 6, 6, 6] )

  cov2 = np.matrix( [[6, 1, 1, 1, 2],
                     [1, 7, 1, 0, 0],
                     [1, 1, 4, 1, 1],
                     [1, 0, 1, 7, 0],
                     [2, 0, 1, 0, 6]] )

  # Create distributions
  d1 = GaussianDistribution(mean1, cov1)
  d2 = GaussianDistribution(mean2, cov2)


  # Set observations to train
  # d1 = 0.4, d2 = 0.6  
  observations = np.zeros((5, 1000))

  for i in range(400):
    observations[:, i] = d1.Random()
  
  for i in range(400, 1000):
    observations[:, i] = d2.Random()
  
  # Convert it to matrix
  observations = np.asmatrix(observations)

  weights = np.array([0.5, 0.5])

  # Create QGMM
  gmm = QuantumGMM(2, 5)
  gmm.dists.append(d1)
  gmm.dists.append(d2)
  gmm.Weights(weights)
  
  # Train QGMM
  gmm.Train(observations, 1)
  #print(gmm.Probability(observations))

  # Check the trained parameters
  print(gmm.weights ** 2)
  print(gmm.dists[0].mean)
  print()
  
  print(gmm.dists[1].mean)
  print()

  print(gmm.dists[0].cov)
  print()

  print(gmm.dists[1].cov)