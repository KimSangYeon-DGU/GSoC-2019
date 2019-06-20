class QuantumGMM:  
  dists = []

  def __init__(self, dist1, dist2):
    print("Initialize QuantumGMM")
    self.dists.append(dist1)
    self.dists.append(dist2)