import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Ellipse

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

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    return width, height, theta

def plot_clustered_data(points, c_means, covs, file_name):
    """Plots the cluster-colored data and the cluster means"""
    colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))
    ax = plt.gca()
    for i in range(points.shape[1]):
      plt.plot(points[:, i][0], points[:, i][1], ".", color="red", zorder=0)
    
    plt.plot(c_means[0][0], c_means[0][1], ".", color="green", zorder=1)
    plt.plot(c_means[1][0], c_means[1][1], ".", color="blue", zorder=1)

    width1, height1, theta1 = cov_ellipse(points, covs[0], nstd=2)
    ellipse1 = Ellipse(xy=(c_means[0][0], c_means[0][1]), width=width1, height=height1, angle=theta1,
                       edgecolor='g', fc='None', lw=2, zorder=4)

    width2, height2, theta2 = cov_ellipse(points, covs[1], nstd=2)                       
    ellipse2 = Ellipse(xy=(c_means[1][0], c_means[1][1]), width=width1, height=height1, angle=theta1,
                       edgecolor='b', fc='None', lw=2, zorder=4)

    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    fig = plt.gcf()
    fig.savefig(file_name)
    plt.close()
    #plt.show()