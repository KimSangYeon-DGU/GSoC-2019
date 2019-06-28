import numpy as np
import sys
import math
from sklearn.cluster import KMeans

class QuantumEMFit:
  maxIterations = 0
  tolerance = 0.0

  def __init__(self, maxIterations, tolerance):
    self.maxIterations = maxIterations
    self.tolerance = tolerance

  def Estimate(self, observations, dists, weights):
    distsTrial = dists
    weightsTrial = weights

    ll = self.LogLikelihood(observations, distsTrial, weightsTrial)

    llOld = -sys.float_info.max

    # Conditional probabilities
    condProbs = np.zeros((observations.shape[1], len(dists)))
    F = np.zeros((observations.shape[1], len(dists)))

    iteration = 1

    while (self.tolerance < abs(ll - llOld) and iteration != self.maxIterations):
    #while (iteration != 20):
      print("QuantumEMFit::Estimate(): iteration {0}, log-likelihood {1}.".format(iteration, ll))

      G1 = distsTrial[0].Probability(observations)
      G2 = distsTrial[1].Probability(observations)

      # Calculate o_{i}
      if np.sum(G1 * G2) != 0:
        o = (G1 * G2) / np.sum(G1 * G2)
      else:
        o = 0
      
      # Calcualte (alpha o)_{i}
      alphao = (1 - weightsTrial[0] ** 2 - weightsTrial[1] ** 2) * o
      #print(np.max(alphao), np.min(alphao))
      commonDivisor = (weightsTrial[0] ** 2) * (G1 ** 2) + (weightsTrial[1] ** 2) * (G2 ** 2) + alphao

      # Calculate Q matrix
      if np.sum(commonDivisor) != 0:
        condProbs[:, 0] = ((weightsTrial[0] ** 2) * (G1 ** 2 ) + 0.5 * alphao) / commonDivisor
        condProbs[:, 1] = ((weightsTrial[1] ** 2) * (G2 ** 2 ) + 0.5 * alphao) / commonDivisor
      else:
        break

      # Calculate F matrix
      F[:, 0] = condProbs[:, 0] - o * np.sum(0.5 * alphao / commonDivisor)
      F[:, 1] = condProbs[:, 1] - o * np.sum(0.5 * alphao / commonDivisor)
      
      # Calculate R matrix
      # Equation 1
      R = 2 * F 

      # Equation 2
      #R = F
      #R[:, 0] += ((weightsTrial[0] ** 2) * (G1 ** 2 )) / ((weightsTrial[0] ** 2) * (G1 ** 2) + (weightsTrial[1] ** 2) * (G2 ** 2) + (0.5 * alphao))
      #R[:, 1] += ((weightsTrial[1] ** 2) * (G2 ** 2 )) / ((weightsTrial[0] ** 2) * (G1 ** 2) + (weightsTrial[1] ** 2) * (G2 ** 2) + (0.5 * alphao))

      # Equation 3
      #R = F
      #R[:, 0] += ((weightsTrial[0] ** 2) * (G1 ** 2 )) / ((weightsTrial[0] ** 2) * (G1 ** 2) + (weightsTrial[1] ** 2) * (G2 ** 2) + alphao)
      #R[:, 1] += ((weightsTrial[1] ** 2) * (G2 ** 2 )) / ((weightsTrial[0] ** 2) * (G1 ** 2) + (weightsTrial[1] ** 2) * (G2 ** 2) + alphao)

      # Normalize row-wise
      for i in range(condProbs.shape[0]):
        probSum = np.sum(condProbs[i])
        
        if (probSum != 0.0):
          condProbs[i] /= probSum

      # Add F by column-wise
      fSums = np.asarray(np.sum(F, 0))
      
      # Add R by column-wise
      rSums = np.asarray(np.sum(R, 0))

      for i in range(len(dists)):
        # Update mean using observations and conditional probabilities
        if fSums[i] != 0:
          distsTrial[i].Mean(np.asarray((np.dot(observations, F[:, i]) / fSums[i]).flatten())[0])
        else:
          continue

        # Update covariance using observations and conditional probabilities
        diffsA = np.transpose(np.transpose(observations) - distsTrial[i].mean)
        
        diffsB = np.multiply(diffsA, np.transpose(R[:, i]))
        cov = np.dot(diffsA, np.transpose(diffsB)) / rSums[i]

        distsTrial[i].Covariance(cov)

      weightsSum = np.sum(weightsTrial)
      weightsTrial = np.sum(condProbs, 0) / observations.shape[1]
      weightsTrial /= weightsSum
      
      # Check the equation of the constraint (20) in the paper
      a = (weightsTrial[0] * G1 - weightsTrial[1] * G2) ** 2
      b = (weightsTrial[0] * G1 + weightsTrial[1] * G2) ** 2
      aMax = np.max(a)
      bMax = np.max(b)
      aMin = np.min(a)
      bMin = np.min(b)
      print(max(aMax, bMax))
      print(max(aMin, bMin))

      llOld = ll

      ll = self.LogLikelihood(observations, distsTrial, weightsTrial)

      iteration += 1
    
    return distsTrial, weightsTrial

  def LogLikelihood(self, observation, dists, weights):
    loglikelihood = 0
    probabilities = None
    likelihoods = np.zeros((len(dists), observation.shape[1]))
    weights /= np.sum(weights)

    for i in range(len(dists)):
      probabilities = dists[i].Probability(observation)
      likelihoods[i] = (weights[i] * probabilities) ** 2
    
    for i in range(len(dists)):
      if np.sum(likelihoods[:, i]) != 0:
        loglikelihood += np.log(np.sum(likelihoods[:, i]))
      else:
        loglikelihood += 0
    
    return loglikelihood

  def InitialClustering(self, observations, dists, weights):
    kmeans = KMeans(n_clusters = len(dists), random_state = 0).fit(np.transpose(observations))
    assignments = kmeans.labels_
    #print(assignments)

    # Create the means and covariances
    means = []
    covs = []
    distsTrial = dists
    weightsTrial = weights
    for i in range(len(dists)):
      means.append(np.zeros(len(dists[0].mean)))
      covs.append(np.zeros((len(dists[0].mean), len(dists[0].mean))))

    # From the assignments, calculate the mean, the covariances, and the weights
    for i in range(observations.shape[1]):
      cluster = assignments[i]
      means[cluster] += np.asarray(observations[:, i].flatten())[0]

      weights[cluster] += 1

    # Normalize the mean
    for i in range(len(dists)):
      if weightsTrial[i] > 1:
        means[i] /= weightsTrial[i]

    for i in range(observations.shape[1]):
      cluster = assignments[i]
      
      diffs = np.asmatrix(np.transpose(observations[:, i]) - means[cluster])[0]
      diffs = np.transpose(diffs)
      covs[cluster] += np.dot(diffs, np.transpose(diffs))

    # Normalize and assign the estimated parameters    
    for i in range(len(dists)):
      if weightsTrial[i] > 1:
        covs[i] /= weightsTrial[i]
      
      distsTrial[i].Mean(means[i])
      distsTrial[i].Covariance(covs[i])

    weightsTrial /= np.sum(weightsTrial)

    return distsTrial, weightsTrial
