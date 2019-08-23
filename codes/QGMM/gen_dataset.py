import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, random
from param_utils import *
def plot_clustered_data(points):
	"""Plots the cluster-colored data and the cluster means"""
	plt.plot(points[0], points[1], ".", color="r", zorder=0)

	fig = plt.gcf()
  
	fig.savefig("./test.png")
	plt.close()


def generate_data():
  df = pd.read_csv('data/multiple_5.csv', sep=',')
  dataset = df.to_numpy()
  dataset = np.transpose(dataset)
  plot_clustered_data(dataset)


def add_circle():
  df = pd.read_csv('data/obs.csv', sep=',')
  dataset = df.to_numpy()
  dataset = np.transpose(dataset)

  x = (1.5 + 4.50) / 2
  y = (5.0 + 0.5) / 2
  
  radius = 1.5
  new_x_list = []
  new_y_list = []
  #radius 3: 500
  #radius 2.5: 500
  #radius 2: 500
  #radius 1.5: 500
  num_pts = int(500 * .67)
  for _ in range(num_pts):
    theta = random.random() * 2 * math.pi
    r = radius * math.sqrt(random.random())
    new_x = r * np.cos(theta) + x
    new_y = r * np.sin(theta) + y
    new_x_list.append(new_x)
    new_y_list.append(new_y)

  ax = plt.gca()
  new_x_list = np.asarray(new_x_list)
  new_y_list = np.asarray(new_y_list)

  new_dataset = []
  new_dataset_x = np.concatenate((dataset[0], new_x_list))
  new_dataset_y = np.concatenate((dataset[1], new_y_list))
  new_dataset.append(new_dataset_x)
  new_dataset.append(new_dataset_y)
  new_dataset = np.asarray(new_dataset)

  plt.plot(new_dataset[0], new_dataset[1], ".", color="r", zorder=0)
  plt.plot(x, y, ".", color="k", zorder=0)
  #plt.plot(new_x_list, new_y_list, ".", color="r", zorder=0)

  #circle = plt.Circle((x, y), radius, color='b', fill=False, linewidth=3)
  #ax.add_patch(circle)

  np.savetxt('data/obs_radius_{0}.csv'.format(radius), new_dataset.T, delimiter=",")

  fig = plt.gcf()
  fig.savefig("./circle_{0}.png".format(radius))
  plt.close()

def mean_range():
  df = pd.read_csv('data/multiple_5.csv', sep=',')
  dataset = df.to_numpy()
  dataset = np.transpose(dataset)
  print(get_initial_means(dataset, 1, 10, 2))

if __name__ == "__main__":  
  #generate_data()
  #add_circle()
  mean_range()