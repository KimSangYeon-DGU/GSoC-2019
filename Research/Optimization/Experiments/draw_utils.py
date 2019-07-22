import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, subprocess, glob
from matplotlib.patches import Ellipse
import matplotlib.animation as animation


CLUSTERS = 2

def eigsorted(cov):
	'''
	Eigenvalues and eigenvectors of the covariance matrix.
	'''
	vals, vecs = np.linalg.eigh(cov)
	order = vals.argsort()[::-1]
	return vals[order], vecs[:, order]

def cov_ellipse(points, cov, nstd):
	"""
	Source: http://stackoverflow.com/a/12321306/1391441
	"""

	vals, vecs = eigsorted(cov)
	theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
	eps = 1e-5
	# Width and height are "full" widths, not radius
	width, height = 2 * nstd * np.sqrt(vals + eps)
	#print("vals: {0}\n cov: {1}".format(vals, cov))
	
	return width, height, theta

def plot_clustered_data(points, c_means, covs, test_name, image_num):
	"""Plots the cluster-colored data and the cluster means"""
	colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))
	ax = plt.gca()
	for i in range(points.shape[1]):
		plt.plot(points[:, i][0], points[:, i][1], ".", color="red", zorder=0)
	
	plt.plot(c_means[0][0], c_means[0][1], ".", color="green", zorder=1)
	plt.plot(c_means[1][0], c_means[1][1], ".", color="blue", zorder=1)

	width1, height1, theta1 = cov_ellipse(points, covs[0], nstd=2)
	ellipse1 = Ellipse(xy=(c_means[0][0], c_means[0][1]), width=width1, \
			height=height1, angle=theta1, edgecolor='g', fc='None', lw=2, zorder=4)
	
	width2, height2, theta2 = cov_ellipse(points, covs[1], nstd=2)
	ellipse2 = Ellipse(xy=(c_means[1][0], c_means[1][1]), width=width2, \
			height=height2, angle=theta2, edgecolor='b', fc='None', lw=2, zorder=4)

	ax.add_patch(ellipse1)
	ax.add_patch(ellipse2)
	
	fig = plt.gcf()
	fig.savefig("./images/{0}/{1:08d}.png".format(test_name, image_num))
	plt.close()

def draw_graph(x, y, x_label, y_label, file_name, test_name):
  print("Save the graph with the file name: {0}".format(file_name))
	# Draw
  plt.plot(x, y, color='r', label=y_label)

	# Set labels
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  fig = plt.gcf()

  # Save
  fig.savefig("./graphs/{0}/".format(test_name)+file_name)

  # Close
  plt.close()

def draw_alphas_graph(x, a1, a2, x_label, y_label, file_name, test_name):
  print("Save the graph with the file name: {0}".format(file_name))
	# Draw
  plt.plot(x, a1, color='g', label='alpha 1')
  plt.plot(x, a2, color='b', label='alpha 2')

	# Set labels
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  fig = plt.gcf()

  # Save
  fig.savefig("./graphs/{0}/".format(test_name)+file_name)

  # Close
  plt.close()

def draw_gaussian(x, g1, g2, x_label, y_label, g1_label, g2_label, file_name, test_name):
  print("Save the graph with the file name: {0}".format(file_name))
	# Draw
  plt.plot(x, g1, color='g', label=g1_label)
  plt.plot(x, g2, color='b', label=g2_label)

	# Set labels
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  fig = plt.gcf()

  # Save
  fig.savefig("./graphs/{0}/".format(test_name)+file_name)

  # Close
  plt.close()


def draw_probs(x, p1, p2, p3, x_label, y_label, file_name, test_name):
  print("Save the graph with the file name: {0}".format(file_name))
	# Draw
  plt.plot(x, p1, color='g', label='P1')
  plt.plot(x, p2, color='b', label='P2')
  plt.plot(x, p3, color='r', label='P1 + P2')

	# Set labels
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  fig = plt.gcf()

  # Save
  fig.savefig("./graphs/{0}/".format(test_name)+file_name)

  # Close
  plt.close()

def generate_video(test_name):
	import cv2
	base_path = './images/{0}'.format(test_name)
	file_names = os.listdir(base_path)
	file_names.sort()
	frame_array = []
	fps = 10
	for file_name in file_names:
		#reading each files
		img = cv2.imread(base_path + '/' + file_name)
		height, width, layers = img.shape
		size = (width,height)
		
		#inserting the frames into an image array
		frame_array.append(img)
	out = cv2.VideoWriter('./videos/{0}.mp4'.format(test_name),\
			cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

	for i in range(len(frame_array)):
		# writing to a image array
		out.write(frame_array[i])
	out.release()

def draw_3d_probability():
	print("test")