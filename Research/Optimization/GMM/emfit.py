import numpy as np
import sys
import math
from sklearn.cluster import KMeans
from draw_utils import *
import pandas as pd

df = pd.read_csv('faithful.csv', sep=',')
observations = df.to_numpy()
observations = np.transpose(observations)

class EMFit:
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

    iteration = 1

    while (self.tolerance < abs(ll - llOld) and iteration != self.maxIterations):
      '''
      mean_list = []
      cov_list = []
      for i in range(len(dists)):
        mean_list.append(distsTrial[i].mean)
        cov_list.append(distsTrial[i].cov)
    
      plot_clustered_data(observations, mean_list, cov_list, "t4", iteration, 2)
      '''

      for i in range(len(dists)):
        probs = distsTrial[i].Probability(observations)
        probs *= weightsTrial[i]
        condProbs[:, i] = probs
      
      print("EMFit::Estimate(): iteration {0}, log-likelihood {1}.".format(iteration, ll))
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

        dists[i].Covariance(cov)

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