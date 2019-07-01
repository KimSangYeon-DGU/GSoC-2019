import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot_learning_curves(X1, X2, C):
  plt.plot(X1, label='x1')
  plt.plot(X2, label='x2')
  plt.plot(C, label='Cost')
  plt.show()

x1 = tf.Variable(initial_value=4, dtype=tf.float32, name='x1')
x2 = tf.Variable(initial_value=-2, dtype=tf.float32, name='x2')

# cost function
J = 40-(x1**2 + x2**2)

optim = tf.train.RMSPropOptimizer(learning_rate=0.01)
training_op = optim.minimize(-J)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

X1, X2, C = [],[],[]
sess.run(init)
n_iter = 2000

for i in range(n_iter):
    _ = sess.run(training_op)
    X1.append(sess.run(x1))
    X2.append(sess.run(x2))
    C.append(sess.run(J))

# Get the final values    
print('after {} iterations:'.format(n_iter))
print('Cost: {} at x1={}, x2={}'.format(J.eval(), x1.eval(), x2.eval() ))
plot_learning_curves(X1, X2, C)

sess.close()