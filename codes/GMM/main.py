from gaussian_distribution import GaussianDistribution
from draw_utils import *
from gmm import GMM
import numpy as np
import pandas as pd
import os, sys, json

if __name__ == "__main__":
  test_name = sys.argv[1]
  json_data = open("jsons/"+test_name+"/"+test_name+".json").read()

  data = json.loads(json_data)

  # Set initial parameters  
  mean1 = np.array(data["means"][0])

  cov1 = np.matrix(data["covs"][0])
  
  mean2 = np.array(data["means"][1])

  cov2 = np.matrix(data["covs"][1])

  # Create distributions
  d1 = GaussianDistribution(mean1, cov1)
  d2 = GaussianDistribution(mean2, cov2)

  df = pd.read_csv("data/"+data["data"], sep=',')
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
  gmm.Train(observations, 1, False)

  means = []; covs = []
  gaussians = data["gaussians"]
  for i in range(gaussians):
    means.append(gmm.dists[i].mean)
    covs.append(gmm.dists[i].cov)
  
  image_path = "./images/" + test_name
  if os.path.exists(image_path) == False:
    os.mkdir(image_path)

  plot_clustered_data(observations, 
    means,
    covs,
    test_name,
    0,
    gaussians)