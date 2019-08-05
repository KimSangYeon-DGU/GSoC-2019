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

def plot_clustered_data(points, c_means, covs, test_name, image_num, gaussians):
	"""Plots the cluster-colored data and the cluster means"""
	#colors = cm.rainbow(np.linspace(0, 1, gaussians))
	colors = ['b', 'g', 'm', 'y', 'c', 'k']

	ax = plt.gca()
	#for i in range(points.shape[1]):
		#plt.plot(points[:, i][0], points[:, i][1], ".", color="r", zorder=0)
	plt.plot(points[0], points[1], ".", color="r", zorder=0)
	
	for i in range(gaussians):
		plt.plot(c_means[i][0], c_means[i][1], ".", color=colors[i], zorder=1)

		width, height, theta = cov_ellipse(points, covs[i], nstd=2)
		ellipse = Ellipse(xy=(c_means[i][0], c_means[i][1]), width=width, \
				height=height, angle=theta, edgecolor=colors[i], fc='None', lw=2, zorder=4)

		ax.add_patch(ellipse)
	
	plt.savefig("./images/{0}/{1:08d}.png".format(test_name, image_num))
	plt.close()

def draw_graph(x, ys, x_label, ys_labels, file_name, test_name):
    print("Save the graph with the file name: {0}".format(file_name))
    colors = ['b', 'g', 'm', 'y', 'c', 'k']
    
    # Draw
    dim = len(ys.shape)
    if dim == 1:
        plt.plot(x, ys, color=colors[0], label=ys_labels[1])
    else:
        n_rows = ys.shape[0]
        for i in range(n_rows):
            plt.plot(x, ys[i], color=colors[i], label=ys_labels[i+1])
    
    plt.grid(True)

    # Set labels
    plt.xlabel(x_label[0])
    plt.ylabel(ys_labels[0])
    plt.legend()
    
    # Save
    plt.savefig("./graphs/{0}/".format(test_name)+file_name)

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