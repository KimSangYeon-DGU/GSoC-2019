import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qgmm_utils import *
from draw_utils import *
import numpy as np
import math


def get_alphas(G, cos):
  alphas = []
  a = tf.math.sqrt(tf.math.abs(1/(2*(cos * tf.reduce_sum(G[0] * G[1]) + 1))))
  alphas.append(a)
  alphas.append(a)
  alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)
  return alphas

def helper_quantum_gmm(observations, G, num_components, phi):
  P = []

  alphas = get_alphas(G, math.cos(phi))

  for i in range(num_components):
    P.append((alphas[i] ** 2) * (G[i] ** 2) + (alphas[0] * alphas[1] \
        * math.cos(phi) * G[0] * G[1]))

  P = tf.convert_to_tensor(P, dtype=tf.float32)  
  
  return P


if __name__ == "__main__":
  x_num = 150
  y_num = 150
  x = np.linspace(-3.8, 3.8, x_num)
  y = np.linspace(-3.8, 3.8, y_num)

  X, Y = np.meshgrid(x, y)
  print(X)
  print(Y)
  # observations
  obs = np.empty(X.shape + (2, ))

  obs[:, :, 0] = X; obs[:, :, 1] = Y
  #print(obs.shape)
  print(obs)
  obs = np.transpose(obs)
  x = np.reshape(obs, (2, x_num * y_num))
  x = tf.convert_to_tensor(x, dtype=tf.float32)

  means = tf.Variable([[-1.5, -1.5], [1.5, 1.5]], \
      dtype=tf.float32, trainable=True)

  covs = tf.Variable([[[1, 0], [0, 1]], \
      [[1, 0], [0, 1]]], dtype=tf.float32, trainable=True)

  G = []
  G.append(unnormalized_gaussians(x, means[0], covs[0], 2))
  G.append(unnormalized_gaussians(x, means[1], covs[1], 2))
  G = tf.convert_to_tensor(G, dtype=tf.float32)

  for degree in range(0, 190, 10):
    print(degree)
    P = helper_quantum_gmm(x, G, 2, np.deg2rad(degree))

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    sess.run(init)

    z = P.eval()
    Z = z[0] + z[1]
    Z = np.reshape(Z, (x_num, y_num))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    jet = plt.get_cmap('jet')

    surf = ax.plot_surface(X, Y, Z, cmap=jet,linewidth=0)
    fig.savefig("{0}.png".format(degree))
