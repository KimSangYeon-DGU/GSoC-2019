import numpy as np
import random
from quantum_emfit import QuantumEMFit

class QuantumGMM:
  gaussians = 0
  dimensionality = 0
  dists = []
  weights = []

  def __init__(self, gaussian, dimensionality):
    self.gaussians = gaussian
    self.dimensionality = dimensionality

  # Calculate phase difference between two distibutions.
  def PhaseDifference(self, observation):
    G1 = self.dists[0].Probability(observation)
    G2 = self.dists[1].Probability(observation)
    
    gaussSum = np.sum(G1 * G2)

    return (1 - self.weights[0] ** 2 - self.weights[1] ** 2) / (2 * self.weights[0] * self.weights[1] * gaussSum)

  def Weights(self, weights):
    self.weights = np.sqrt(weights)

  # Warning! the observation should be matrix format.
  def Probability(self, observation, component = -1):
    G1 = self.dists[0].Probability(observation)
    G2 = self.dists[1].Probability(observation)

    P1 = (self.weights[0] ** 2) * (G1 ** 2) + self.weights[0] * self.weights[1] * G1 * G2 * self.PhaseDifference(observation)
    P2 = (self.weights[1] ** 2) * (G2 ** 2) + self.weights[0] * self.weights[1] * G1 * G2 * self.PhaseDifference(observation)

    if component == 0:
      return P1
    elif component == 1:
      return P2
    else:
      return P1 * P2

  def Random(self):
    threshold = random.random()
    gaussian = 0

    sumProb = 0

    for g in range(self.gaussians):
      sumProb += (self.weights[g] ** 2)
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
      likelihoods[i] = (weights[i] * probabilities) ** 2
    
    for i in range(self.gaussians):
      loglikelihood += np.log(np.sum(likelihoods[:, i]))
    
    return loglikelihood

  def Train(self, observation, trials):
    fitter = QuantumEMFit(300, 1e-10)
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
