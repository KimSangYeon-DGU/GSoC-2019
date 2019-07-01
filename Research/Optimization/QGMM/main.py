from quantum_gaussian_distribution import QuantumGaussianDistribution
from quantum_gmm import QuantumGMM
from gaussian_distribution import GaussianDistribution
from gmm import GMM
import numpy as np

if __name__ == "__main__":
  
  # Set initial parameters (hard case)
  '''
  mean1 = np.array( [5, 6, 3, 3, 2] )

  cov1 = np.matrix( [[6, 1, 1, 1, 2],
                     [1, 7, 1, 0, 0],
                     [1, 1, 4, 1, 1],
                     [1, 0, 1, 7, 0],
                     [2, 0, 1, 0, 6]] )

  mean2 = np.array( [3, 3, 3, 3, 3] )

  cov2 = np.matrix( [[6, 1, 1, 1, 2],
                     [1, 7, 1, 0, 0],
                     [1, 1, 4, 1, 1],
                     [1, 0, 1, 7, 0],
                     [2, 0, 1, 0, 6]] )
  '''

  # Set initial parameters (easy case)
  mean1 = np.array( [5, 6, 3, 3, 2] )

  cov1 = np.matrix( [[6, 1, 1, 1, 2],
                     [1, 7, 1, 0, 0],
                     [1, 1, 4, 1, 1],
                     [1, 0, 1, 7, 0],
                     [2, 0, 1, 0, 6]] )

  mean2 = np.array( [1, -1, 0, 1, 1] )
  #mean2 = np.array( [4, 5, 2, 2, 1] )
  '''
  cov2 = np.matrix( [[5, 0, 0, 1, 1],
                     [0, 7, 0, 0, 0],
                     [0, 0, 2, 0, 0],
                     [1, 0, 0, 3, 0],
                     [1, 0, 0, 0, 5]] )

  '''
  cov2 = np.matrix( [[1, 0, 0, 1, 1],
                     [0, 2, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [1, 0, 0, 1, 0],
                     [1, 0, 0, 0, 1]] )

  dummyMean = np.zeros(len(mean1))
  dummyCov = np.eye((len(mean1)))

  # Create distributions for observation generations
  d1 = GaussianDistribution(mean1, cov1)
  d2 = GaussianDistribution(mean2, cov2)

  # Create distributions for consisting of QGMM.
  d3 = QuantumGaussianDistribution(dummyMean, dummyCov)
  d4 = QuantumGaussianDistribution(dummyMean, dummyCov)

  # Create distribution for consisting of GMM.
  d5 = GaussianDistribution(dummyMean, dummyCov)
  d6 = GaussianDistribution(dummyMean, dummyCov)

  # Set observations to train
  #totalObsNum = 5000
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
  #print(observations)
  
  
  # Create QGMM
  qgmm = QuantumGMM(2, 5)
  qgmm.dists.append(d3)
  qgmm.dists.append(d4)
  qgmm.Weights(weights)
  
  # Train QGMM
  qgmm.Train(observations, 3)

  
  # Create QGMM
  gmm = GMM(2, 5)
  gmm.dists.append(d5)
  gmm.dists.append(d6)
  gmm.weights = weights
  
  # Train QGMM
  #gmm.Train(observations, 3)
  
  
  # Check the trained parameters
  sortedIndices = np.argsort(qgmm.weights)
  print("*** Check the trained parameters of QGMM ***")
  print("d1's weight: {0}".format(qgmm.weights[sortedIndices[0]]))
  print("d2's weight: {0}\n".format(qgmm.weights[sortedIndices[1]]))

  print("d1's mean: {0}".format(qgmm.dists[sortedIndices[0]].mean))
  print("d2's mean: {0}\n".format(qgmm.dists[sortedIndices[1]].mean))

  print("d1's covariance: \n{0}\n".format(qgmm.dists[sortedIndices[0]].cov))
  print("d2's covariance: \n{0}\n".format(qgmm.dists[sortedIndices[1]].cov))
  
  '''
  # Check the trained parameters
  sortedIndices = np.argsort(gmm.weights)
  print("*** Check the trained parameters of GMM ***")
  print("d1's weight: {0}".format(gmm.weights[sortedIndices[0]]))
  print("d2's weight: {0}\n".format(gmm.weights[sortedIndices[1]]))

  print("d1's mean: {0}".format(gmm.dists[sortedIndices[0]].mean))
  print("d2's mean: {0}\n".format(gmm.dists[sortedIndices[1]].mean))

  print("d1's covariance: \n{0}\n".format(gmm.dists[sortedIndices[0]].cov))
  print("d2's covariance: \n{0}\n".format(gmm.dists[sortedIndices[1]].cov))
  '''
