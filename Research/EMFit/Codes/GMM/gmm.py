import numpy as np

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
  
  #def Random():
