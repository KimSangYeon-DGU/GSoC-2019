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

    # Make directories
    os.mkdir("./graphs/{0}".format(test_name))
    os.mkdir("./csvs/{0}".format(test_name))
    os.mkdir("./images/{0}".format(test_name))

    # Initialize means and covariances.
    dimensionality = 2

    # Set the number of Gaussians
    gaussians = 2

    alphas = tf.Variable([0.5, 0.5], dtype=tf.float32, trainable=True)

    '''
    m1, m2 = get_initial_means(dataset)
    print(m1, m2)

    means = tf.Variable([[m1[0], m1[1]], [m2[0], m2[1]]], \
        dtype=tf.float32, trainable=True)
    '''
    means = tf.Variable([[_means1[0], _means1[1]], [_means2[0], _means2[1]]], \
        dtype=tf.float32, trainable=True)

    phis = tf.Variable([_phis[0], _phis[1]], \
        dtype=tf.float32, trainable=True)

    covs = tf.Variable([[[0.08, 0.1],
                        [0.1, 3.3]], \
                            
                        [[0.08, 0.1],
                        [0.1, 3.3]]], dtype=tf.float32, trainable=True)

    # Calculate normalized gaussians
    G = []
    for i in range(gaussians):
        G.append(unnormalized_gaussians(obs, means[i], covs[i], dimensionality))

    G = tf.convert_to_tensor(G, dtype=tf.float32)

    P = quantum_gmm(obs, G, alphas, gaussians, phis)

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

        return tf.reduce_sum(log_likelihood)

    def approx_constraint(G, alphas, phis, gaussians):
        mix_sum = 0
        for k in range(gaussians):
            for l in range(gaussians):
                if k == l:
                    continue
                mix_sum += alphas[k] * alphas[l] * G[k] * G[l] * get_cosine(phis, k, l)
        return tf.math.abs(tf.reduce_sum(alphas ** 2) + tf.reduce_sum(mix_sum) - 1)

    # Objective function
    J = -loglikeihood(Q, P, gaussians) + ld * approx_constraint(G, alphas, phis, gaussians)

    # Set optimizer to Adam with learning rate 0.01
    optim = tf.train.AdamOptimizer(learning_rate=lr)
    training = optim.minimize(J)

    # Build session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    sess.run(init)

    # Set the number of iterations is 2000.
    n_iter = 100

    plot_clustered_data(dataset, means.eval(), covs.eval(), test_name, 0, gaussians)

    # For graph
    max_iteration = 500

    x_records = []
    nll_records = []
    constraint_records = []
    alpha_records = []
    phi_records = []
    gauss_records = []
    prob_records = []
    j_records = []

    # Save initial values
    x_records.append(0)
    nll_records.append(-loglikeihood(Q, P, gaussians).eval())
    constraint_records.append(approx_constraint(G, alphas, phis, gaussians).eval())
    alpha_records.append(alphas.eval())
    phi_records.append(phis.eval())
    gauss_records.append(tf.reduce_sum(G, axis=1).eval())
    prob_records.append(tf.reduce_sum(P, axis=1).eval())

    tot = 1e-3
    sess.run(J)
    cur_J = J.eval()
    pre_J = cur_J

    j_records.append(cur_J)

    def optimize():
        for _ in range(n_iter):
            sess.run(training)

    # Train QGMM
    for i in range(1, max_iteration):
        print(i, test_name, tf.reduce_sum(P[0] + P[1]).eval())
        print(phis.eval(), cur_J)
        optimize()
        cur_J = J.eval()
        cur_alphas = alphas.eval()
        cur_means = means.eval()
        cur_covs = covs.eval()

        plot_clustered_data(dataset, cur_means, cur_covs, test_name, i, gaussians)

        # Save values for graphs
        x_records.append(i * n_iter)
        nll_records.append(-loglikeihood(Q, P, gaussians).eval())
        constraint_records.append(approx_constraint(G, alphas, phis, gaussians).eval())
        alpha_records.append(cur_alphas)
        phi_records.append(phis.eval())
        gauss_records.append(tf.reduce_sum(G, axis=1).eval())
        prob_records.append(tf.reduce_sum(P, axis=1).eval())
        j_records.append(cur_J)

        if abs(pre_J - cur_J) < tot:
            break

        pre_J = cur_J

    # Check the trained parameters with actual mean and covariance using numpy
    print('\nCost:{0}\n\nalphas:\n{1}\n\nmeans:\n{2}\n\ncovariances:\n{3}\n\n'.\
        format(cur_J, cur_alphas, cur_means, cur_covs))

    # Save data to csv format and a graph
    record_csv_graph(x_records,
                     nll_records,
                     constraint_records,
                     alpha_records,
                     phi_records,
                     gauss_records,
                     prob_records,
                     j_records,
                     test_name)

    # Generate a video
    generate_video(test_name)

    sess.close()

if __name__== "__main__":
    n_tests = len(test_cases)

    for i in range(n_tests):
        if test_cases[i]["run"] == False:
            continue
        train_qgmm(test_cases[i]["name"], test_cases[i]["mean1"], 
                   test_cases[i]["mean2"], test_cases[i]["ld"],
                   test_cases[i]["phis"])
