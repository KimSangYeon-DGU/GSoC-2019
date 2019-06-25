import numpy as np
import sys
import math

class QuantumEMFit:
  maxIterations = 0
  tolerance = 0.0

  def __init__(self, maxIterations, tolerance):
    self.maxIterations = maxIterations
    self.tolerance = tolerance

  def Estimate(self, observations, dists, weights):
    distsTrial = dists
    weightsTrial = weights

    # If we use the clustering algorithm like K-Means initially, the performance would be increased.
    # However, this time, I skip any initial clustering.
    #distsTrial, weightsTrial = self.InitialClustering(distsTrial, weightsTrial)

    ll = self.LogLikelihood(observations, distsTrial, weightsTrial)

    llOld = -sys.float_info.max

    # Conditional probabilities
    condProbs = np.zeros((observations.shape[1], len(dists)))
    F = np.zeros((observations.shape[1], len(dists)))

    iteration = 1

    while (self.tolerance < abs(ll - llOld) and iteration != self.maxIterations):
      print("EMFit::Estimate(): iteration {0}, log-likelihood {1}.".format(iteration, ll))

      G1 = distsTrial[0].Probability(observations)
      G2 = distsTrial[1].Probability(observations)

      # Calculate o_{i}
      if np.sum(G1 * G2) == 0:
        break

      o = (G1 * G2) / np.sum(G1 * G2)
      
      # Calcualte (alpha o)_{i}
      alphao = (1 - weightsTrial[0] ** 2 - weightsTrial[1] ** 2) * o

      commonDivisor = (weightsTrial[0] ** 2) * (G1 ** 2) + (weightsTrial[1] ** 2) * (G2 ** 2) + alphao

      # Calculate Q matrix
      condProbs[:, 0] = ((weightsTrial[0] ** 2) * (G1 ** 2 ) + 0.5 * alphao) / commonDivisor
      condProbs[:, 1] = ((weightsTrial[1] ** 2) * (G2 ** 2 ) + 0.5 * alphao) / commonDivisor

      # Calculate F matrix
      F[:, 0] = condProbs[:, 0] - o * np.sum(0.5 * alphao / commonDivisor)
      F[:, 1] = condProbs[:, 1] - o * np.sum(0.5 * alphao / commonDivisor)
      
      # Calculate R matrix
      R = 2 * F

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

      weightsTrial = np.sum(condProbs, 0) / observations.shape[1]
      llOld = ll

      ll = self.LogLikelihood(observations, distsTrial, weightsTrial)

      iteration += 1

    
    return distsTrial, weightsTrial

  def LogLikelihood(self, observation, dists, weights):
    loglikelihood = 0
    probabilities = None
    likelihoods = np.zeros((len(dists), observation.shape[1]))

    for i in range(len(dists)):
      probabilities = dists[i].Probability(observation)
      likelihoods[i] = (weights[i] * probabilities) ** 2
    
    for i in range(len(dists)):
      loglikelihood += np.log(np.sum(likelihoods[:, i]))
    
    return loglikelihood

  def InitialClustering(self, dists, weights):
    print("Initial Clustering")