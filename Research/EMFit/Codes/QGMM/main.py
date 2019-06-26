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
  totalObsNum = 1000
  w1 = 0.3
  w2 = 1 - w1
  observations = np.zeros((5, totalObsNum))

  for i in range(int(w1 * totalObsNum)):
    observations[:, i] = d1.Random()
  
  for i in range(int(w1 * totalObsNum), totalObsNum):
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
  gmm.Train(observations, 3)
  #print(gmm.Probability(observations))

  # Check the trained parameters
  print("*** Check the trained parameters ***")
  print("d1's weight: {0}".format(gmm.weights[0] ** 2))
  print("d2's weight: {0}\n".format(gmm.weights[1] ** 2))

  print("d1's mean: {0}".format(gmm.dists[0].mean))
  print("d2's mean: {0}\n".format(gmm.dists[1].mean))

  print("d1's covariance: \n{0}\n".format(gmm.dists[0].cov))
  print("d2's covariance: \n{0}\n".format(gmm.dists[1].cov))