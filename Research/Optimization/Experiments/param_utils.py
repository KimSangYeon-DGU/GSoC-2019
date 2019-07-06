import random

def get_initial_means():
  means = []
  means.append(random.uniform(1, 5)) # X
  means.append(random.uniform(30, 100)) # Y

  return means