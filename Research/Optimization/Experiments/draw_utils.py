import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, subprocess, glob
from matplotlib.patches import Ellipse
from qgmm_utils import compose_covariance
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

def plot_clustered_data(points, c_means, covs, file_name, image_num):
	"""Plots the cluster-colored data and the cluster means"""
	colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))
	ax = plt.gca()
	for i in range(points.shape[1]):
		plt.plot(points[:, i][0], points[:, i][1], ".", color="red", zorder=0)
	
	plt.plot(c_means[0][0], c_means[0][1], ".", color="green", zorder=1)
	plt.plot(c_means[1][0], c_means[1][1], ".", color="blue", zorder=1)

	cov_lower = np.tril(covs[0])
	width1, height1, theta1 = cov_ellipse(points, np.dot(cov_lower, np.transpose(cov_lower)), nstd=2)
	ellipse1 = Ellipse(xy=(c_means[0][0], c_means[0][1]), width=width1, \
			height=height1, angle=theta1, edgecolor='g', fc='None', lw=2, zorder=4)
	
	cov_lower = np.tril(covs[1])
	width2, height2, theta2 = cov_ellipse(points, np.dot(cov_lower, np.transpose(cov_lower)), nstd=2)
	ellipse2 = Ellipse(xy=(c_means[1][0], c_means[1][1]), width=width2, \
			height=height2, angle=theta2, edgecolor='b', fc='None', lw=2, zorder=4)

	ax.add_patch(ellipse1)
	ax.add_patch(ellipse2)
	
	fig = plt.gcf()
	fig.savefig("./videos/file{0:08d}.png".format(image_num))
	plt.close()
    
def generate_video():        
	os.chdir("./videos")
	subprocess.call([
			'ffmpeg', '-framerate', '8', '-i', 'file%08d.png', '-r', '30', '-pix_fmt', 'yuv420p',
			'video_name.mp4'
	])
	for file_name in glob.glob("*.png"):
			os.remove(file_name)

def draw_graph(x, y, x_label, y_label, file_name):
  print("Save the graph with the file name: {0}".format(file_name))
	# Draw
  plt.plot(x, y, color='r', label=y_label)

	# Set labels
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  fig = plt.gcf()

  # Save
  fig.savefig("./graphs/"+file_name)

  # Close
  plt.close()

def draw_alphas_graph(x, a1, a2, x_label, y_label, file_name):
  print("Save the graph with the file name: {0}".format(file_name))
	# Draw
  plt.plot(x, a1, color='r', label='alpha 1')
  plt.plot(x, a2, color='b', label='alpha 2')

	# Set labels
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  fig = plt.gcf()

  # Save
  fig.savefig("./graphs/"+file_name)

  # Close
  plt.close()


def generate_video2():
	import cv2
	base_path = './videos'
	file_names = os.listdir(base_path)
	file_names.sort()
	frame_array = []
	fps = 20
	for file_name in file_names:
		#reading each files
		img = cv2.imread(base_path + '/' + file_name)
		height, width, layers = img.shape
		size = (width,height)
		
		#inserting the frames into an image array
		frame_array.append(img)
	out = cv2.VideoWriter('./video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	for i in range(len(frame_array)):
		# writing to a image array
		out.write(frame_array[i])
	out.release()