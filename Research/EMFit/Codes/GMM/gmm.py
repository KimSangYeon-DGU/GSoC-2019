import numpy as np
import random
from emfit import EMFit

class GMM:
  gaussians = 0
  dimensionality = 0
  dists = []
  weights = []

  def __init__(self, gaussian, dimensionality):
    self.gaussians = gaussian
    self.dimensionality = dimensionality

  def Probability(self, observation, component = -1):
    if component < 0:
      return np.exp(self.LogProbability(observation))
    else:
      return np.exp(self.LogProbability(observation, component))

  def LogProbability(self, observation, component = -1):
    if component < 0:
      sum = 0
      for i in range(self.gaussians):
        sum += np.log(self.weights[i]) + self.dists[i].LogProbability(observation)

      return sum
    else:
      return np.log(self.weights[component]) + self.dists[component].LogProbability(observation)
  
  def Random(self):
    threshold = random.random()
    gaussian = 0

    sumProb = 0

    for g in range(self.gaussians):
      sumProb += self.weights[g]
      if threshold <= sumProb:
        gaussian = g
        break

    return self.dists[gaussian].Random()

  def LogLikelihood(self, observation, dists, weights):
    loglikelihood = 0
    probabilities = None
    likelihoods = np.zeros((self.gaussians, observation.shape[1]))

    for i in range(self.gaussians):
      probabilities = dists[i].Probability(observation)
      likelihoods[i] = weights[i] * probabilities
    
    for i in range(self.gaussians):
      loglikelihood += np.log(np.sum(likelihoods[:, i]))
    
    return loglikelihood

  def Train(self, observation, trials):
    fitter = EMFit(300, 1e-10)
    self.dists, self.weights = fitter.Estimate(observation, self.dists, self.weights)

    bestLikelihood = self.LogLikelihood(observation, self.dists, self.weights)

    print("GMM::Train(): Log-likelihood of trial 1 is {0}.".format(bestLikelihood))

    distsTrial = self.dists
    weightsTrial = self.weights

    for i in range(1, trials):
      distsTrial, weightsTrial = fitter.Estimate(observation, distsTrial, weightsTrial)
      newLikelihood = self.LogLikelihood(observation, distsTrial, weightsTrial)

      print("GMM::Train(): Log-likelihood of trial {0} is {1}.".format(i + 1, newLikelihood))

      if bestLikelihood < newLikelihood:
        bestLikelihood = newLikelihood

        self.dists = distsTrial
        self.weights = weightsTrial
    
    print("GMM::Train(): Log-likelihood of trained GMM is {0}.".format(bestLikelihood))

    return bestLikelihood
