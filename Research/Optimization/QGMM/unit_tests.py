import numpy as np
import tensorflow as tf
from qgmm_utils import *
import math

def unnormalized_gaussians_single_observation_test():
  means = tf.Variable([ [5, 6, 3, 3, 2], [0, 0, 1, 1, 1] ], dtype=tf.float32, trainable=True)
  covs = tf.Variable([ [ [6, 1, 1, 1, 2],
                         [1, 7, 1, 0, 0],
                         [1, 1, 4, 1, 1],
                         [1, 0, 1, 7, 0],
                         [2, 0, 1, 0, 6] ], 
                         
                       [ [6, 1, 1, 1, 2],
                         [1, 7, 1, 0, 0],
                         [1, 1, 4, 1, 1],
                         [1, 0, 1, 7, 0],
                         [2, 0, 1, 0, 6] ]], dtype=tf.float32, trainable=True)

  obs = tf.constant([[0], 
                     [1],
                     [2],
                     [3],
                     [4]], dtype=tf.float32)

  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()

  sess.run(init)

  Gs = unnormalized_gaussians(obs, means[0], covs[0], dimensionality=2)
  print(Gs.eval()**2)

def unnormalized_gaussians_multi_observations_test():
  means = tf.Variable([ [5, 6, 3, 3, 2], [0, 0, 1, 1, 1] ], dtype=tf.float32, trainable=True)
  covs = tf.Variable([ [ [6, 1, 1, 1, 2],
                         [1, 7, 1, 0, 0],
                         [1, 1, 4, 1, 1],
                         [1, 0, 1, 7, 0],
                         [2, 0, 1, 0, 6] ], 
                         
                       [ [6, 1, 1, 1, 2],
                         [1, 7, 1, 0, 0],
                         [1, 1, 4, 1, 1],
                         [1, 0, 1, 7, 0],
                         [2, 0, 1, 0, 6] ]], dtype=tf.float32, trainable=True)

  obs = tf.constant([[0, 3, 2, 2, 3, 4], 
                     [1, 2, 2, 1, 0, 0],
                     [2, 3, 0, 5, 5, 6],
                     [3, 7, 8, 0, 1, 1],
                     [4, 8, 1, 1, 0, 0]], dtype=tf.float32)

  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()

  sess.run(init)
  
  Gs = unnormalized_gaussians(obs, means[0], covs[0], dimensionality=2)
  print(Gs.eval())


def apply_positive_definite_constraint_test():
  cov = tf.Variable([ [ [6, -100, 1, 1, 2],
                         [-100, 7, 1, 0, 0],
                         [1, 1, 4, 1, 1],
                         [1, 0, 1, 7, 0],
                         [2, 0, 1, 0, 6] ] ], dtype=tf.float32, trainable=True)
  
  c = apply_positive_definite_constraint(cov[0], 5)
  decomposed_c = tf.linalg.cholesky(c)

  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()
  sess.run(init) 

  print(c.eval())
  print(decomposed_c.eval())


def simple_test():
  sum_ = 0
  tensors = tf.Variable([[1, 2, 3], [3, 4, 5], [2, 5, 6]], dtype=tf.float32)
  for i in range(3):
    sum_ += tensors[i]
  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()
  sess.run(init) 
  print(sum_.eval())

if __name__ == "__main__":
  #unnormalized_gaussians_single_observation_test()
  #unnormalized_gaussians_multi_observations_test()
  #apply_positive_definite_constraint_test()
  simple_test()