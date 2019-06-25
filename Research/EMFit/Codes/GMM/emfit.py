import numpy as np
import sys
import math

class EMFit:
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

    iteration = 1

    while (self.tolerance < abs(ll - llOld) and iteration != self.maxIterations):
      print("EMFit::Estimate(): iteration {0}, log-likelihood {1}.".format(iteration, ll))

      for i in range(len(dists)):
        probs = distsTrial[i].Probability(observations)
        probs *= weightsTrial[i]
        condProbs[:, i] = probs

      # Normalize row-wise
      for i in range(condProbs.shape[0]):
        probSum = np.sum(condProbs[i])
        
        if (probSum != 0.0):
          condProbs[i] /= probSum

      # Add condProbs by column-wise
      probRowSums = np.asarray(np.sum(condProbs, 0))
      

      for i in range(len(dists)):
        # Update mean using observations and conditional probabilities
        if probRowSums[i] != 0:
          distsTrial[i].Mean(np.asarray((np.dot(observations, condProbs[:, i]) / probRowSums[i]).flatten())[0])
        else:
          continue

        # Update covariance using observations and conditional probabilities
        diffsA = np.transpose(np.transpose(observations) - distsTrial[i].mean)
        

        diffsB = np.multiply(diffsA, np.transpose(condProbs[:, i]))
        cov = np.dot(diffsA, np.transpose(diffsB)) / probRowSums[i]

        distsTrial[i].Covariance(cov)

      weightsTrial = probRowSums / observations.shape[1]
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
      likelihoods[i] = weights[i] * probabilities
    
    for i in range(len(dists)):
      loglikelihood += np.log(np.sum(likelihoods[:, i]))
    
    return loglikelihood

  def InitialClustering(self, dists, weights):
    print("Initial Clustering")