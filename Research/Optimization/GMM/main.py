from gaussian_distribution import GaussianDistribution
from gmm import GMM
import numpy as np
import pandas as pd

if __name__ == "__main__":

  # Set initial parameters
  #mean1 = np.array( [2.6585135519388348, 54.66062219876824] )
  #mean1 = np.array( [3.427976229515216, 61.46413393088303] )
  #mean1 = np.array( [2.756031811312966, 76.62447648112042] )
  mean1 = np.array( [4.893025788130122, 59.46713813379837] )
  #mean1 = np.array( [4.171021823127277, 83.66322004888708] )

  cov1 = np.matrix( [[0.08, 0.1],
                     [0.1, 3.3]] )

  #mean2 = np.array( [3.1085745233652995, 77.99698134521407] )
  #mean2 = np.array( [4.5517041217554945, 51.756595162050985] )
  #mean2 = np.array( [2.9226572802266397, 88.3509418943818] )
  mean2 = np.array( [2.080000263954121, 78.15976694366192] )
  #mean2 = np.array( [1.781079954983019, 95.411542531776] )

  cov2 = np.matrix( [[0.08, 0.1],
                     [0.1, 3.3]] )

  # Create distributions
  d1 = GaussianDistribution(mean1, cov1)
  d2 = GaussianDistribution(mean2, cov2)

  df = pd.read_csv('faithful.csv', sep=',')
  observations = df.to_numpy()
  observations = np.transpose(observations)
  
  # Convert it to matrix
  observations = np.asmatrix(observations)

  # Create GMM
  gmm = GMM(2, 2)
  gmm.dists.append(d1)
  gmm.dists.append(d2)
  gmm.weights = [0.5, 0.5]
  
  # Train GMM
  #gmm.Train(observations, 1, False)
  gmm.Train(observations, 1, False)

  # Check the trained parameters
  print(gmm.weights)
  print(gmm.dists[0].mean)
  print()
  
  print(gmm.dists[1].mean)
  print()

  print(gmm.dists[0].cov)
  print()

  print(gmm.dists[1].cov)