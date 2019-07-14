import random
import numpy as np

def get_initial_means():
  means = []
  means.append(random.uniform(2, 5)) # X
  means.append(random.uniform(40, 80)) # Y

  return means
'''
def get_initial_means_from_dataset(dataset):
  x = 0.0; y = 0.0
  num = 5
  for i in range(num):
    rand = int(random.uniform(0, 270))
    x += dataset[0][rand]
    y += dataset[1][rand]
  print(rand)

  means = []
  means.append(x / num) # X
  means.append(y / num) # Y

  return means
'''
def get_initial_means_from_dataset(dataset):
  # Set means1
  rand = int(random.uniform(0, 270))
  means1 = []
  x1 = dataset[0][rand]
  y1 = dataset[1][rand]
  means1.append(x1) # X
  means1.append(y1) # Y

  # Set means2
  means2 = []
  if 3.5 < x1:
    x2 = 3.5 - (x1 - 3.5)
  else:
    x2 = 3.5 + (3.5 - x1)
  
  if 70 < y1:
    y2 = 70 - (y1 - 70)
  else:
    y2 = 70 + (70 - y1)

  means2.append(x2) # X
  means2.append(y2) # Y

  return means1, means2