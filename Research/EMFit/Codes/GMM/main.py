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

  mean2 = np.array( [2, 3, 3, 1, 2] )
  cov2 = np.matrix( [[0, 2, 3, 2, 1],
                     [2, 0, 3, 2, 1],
                     [1, 2, 0, 2, 4],
                     [0, 3, 1, 0, 3],
                     [1, 3, 2, 5, 0]] )

  # Set observations to train
  observations = np.matrix( [[0, 3, 2, 2, 3, 4],
                             [1, 2, 2, 1, 0, 0],
                             [2, 3, 0, 5, 5, 6],
                             [3, 7, 8, 0, 1, 1],
                             [4, 8, 1, 1, 0, 0]] )

  #print(observations[:, 0])
  # Create distributions
  d1 = GaussianDistribution(mean1, cov1)
  d2 = GaussianDistribution(mean2, cov2)

  # Create GMM
  