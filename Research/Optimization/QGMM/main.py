from qgmm_utils import *
from draw_utils import *
from param_utils import *
from test_utils import *
from record_utils import *
import numpy as np
import tensorflow as tf
import pandas as pd
import math, os

def train_qgmm(_test_name, _means1, _means2, _ld, _phis):
    # Load 'Old faithful' dataset
    df = pd.read_csv('faithful.csv', sep=',')
    dataset = df.to_numpy()
    dataset = np.transpose(dataset)
    obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

    test_name = "{0}_{1}_{2}".format(_test_name, _phis[0]-_phis[1], _ld)
    
    images_path = "images/{0}".format(test_name)

    if os.path.exists(images_path) == False:
        os.mkdir("images/{0}".format(images_path))

    # Initialize means and covariances.
    dimensionality = 2

    # Set the number of Gaussians
    gaussians = 2

    alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True, name="alphas")

    '''
    m1, m2 = get_initial_means(dataset)
    print(m1, m2)

    means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
        dtype=tf.float32, trainable=True)
    '''
    means = tf.Variable([[_means1[0], _means1[1]], [_means2[0], _means2[1]]], \
        dtype=tf.float32, trainable=True, name="means")

    phis = tf.Variable([_phis[0], _phis[1]], \
        dtype=tf.float32, trainable=True, name="phis")

    covs = tf.Variable([[[0.08, 0.1],
                        [0.1, 3.3]], \
                            
                        [[0.08, 0.1],
                        [0.1, 3.3]]], dtype=tf.float32, trainable=True, name="covs")

    # Calculate normalized gaussians
    G = []
    for i in range(gaussians):
        G.append(unnormalized_gaussians(obs, means[i], covs[i], dimensionality))

    G = tf.convert_to_tensor(G, dtype=tf.float32, name="G")

    P = quantum_gmm(obs, G, alphas, gaussians, phis)
    P = tf.convert_to_tensor(P, dtype=tf.float32, name="P")

    Q = get_Q(P, gaussians); Q = tf.stop_gradient(Q)

    # lambda
    ld = _ld

    # learning rate
    lr = 0.001

    # Objective function :: Minimize (NLL + lambda * approximation constant)
    # Approximation constant :: (Sum of P(p_{i}, k| alpha_{k}, theta_{k})) - 1 = 0
    def loglikeihood(Q, P, gaussians):
        log_likelihood = 0
        for k in range(gaussians):
            log_likelihood += Q[k] * tf.math.log(tf.clip_by_value(P[k], 1e-10, 1e10))

        return tf.reduce_sum(log_likelihood, name="ll")

    def approx_constraint(G, alphas, phis, gaussians):
        mix_sum = 0
        for k in range(gaussians):
            for l in range(gaussians):
                if k == l:
                    continue
                mix_sum += alphas[k] * alphas[l] * G[k] * G[l] * get_cosine(phis, k, l)
        return tf.math.abs(tf.reduce_sum(alphas ** 2) + tf.reduce_sum(mix_sum) - 1, name="constraint")

    # Objective function
    J = -loglikeihood(Q, P, gaussians) + ld * approx_constraint(G, alphas, phis, gaussians)

    # Set optimizer to Adam with learning rate 0.01
    optim = tf.train.AdamOptimizer(learning_rate=lr)
    training = optim.minimize(J)

    # Build session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    sess.run(init)

    plot_clustered_data(dataset, means.eval(), covs.eval(), test_name, 0, gaussians)

    # For graph
    max_iteration = 5000

    tot = 1e-3

    saver = tf.train.Saver(max_to_keep=10000)
    saver.save(sess, "models/{0}/{1}.ckpt".format(test_name, 0))
    cur_J = sess.run(J)
    pre_J = cur_J

    # Train QGMM
    for i in range(1, max_iteration):
        sess.run(training)
        
        if i % 100 == 0:
            print(i, test_name)
            cur_J = sess.run(J)
            
            if abs(pre_J - cur_J) < tot:
                break

            pre_J = cur_J

            saver.save(sess, "models/{0}/{1}.ckpt".format(test_name, i), write_meta_graph=False)
            print(means)
    
    saver.save(sess, "models/{0}/{1}.ckpt".format(test_name, i), write_meta_graph=False)
    sess.close()

if __name__== "__main__":
    n_tests = len(test_cases)

    for i in range(n_tests):
        if test_cases[i]["run"] == False:
            continue
        train_qgmm(test_cases[i]["name"], test_cases[i]["mean1"], 
                   test_cases[i]["mean2"], test_cases[i]["ld"],
                   test_cases[i]["phis"])
