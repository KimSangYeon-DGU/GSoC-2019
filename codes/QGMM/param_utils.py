import random
import numpy as np

def get_initial_means(dataset, x_offset, y_offset, gaussians):
  x_min = np.min(dataset[0])
  x_max = np.max(dataset[0])
  y_min = np.min(dataset[1])
  y_max = np.max(dataset[1])

  means = []
  for i in range(gaussians):
    x_rand = random.uniform(x_min, x_max)
    y_rand = random.uniform(y_min, y_max)
    x_offs = random.uniform(-x_offset, x_offset)
    y_offs = random.uniform(-y_offset, y_offset)

    means.append([x_rand + x_offs, y_rand + y_offs])
  
  means = np.asarray(means)
  return means
