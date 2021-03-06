from qgmm_utils import *
from draw_utils import *
from param_utils import *
import numpy as np
import tensorflow as tf
import pandas as pd
import math, os, json, sys

def train_qgmm(_means, _covs, _alphas, _phis, _ld, _data, 
							 _gaussians, _dimensionality,_test_name):
	# Load 'Old faithful' dataset
	df = pd.read_csv('data/{0}'.format(_data), sep=',')
	dataset = df.to_numpy(); dataset = np.transpose(dataset)
	obs = tf.convert_to_tensor(dataset, dtype=tf.float32)

	test_name = "{0}_{1}_{2}".format(_test_name, int(_phis[0]-_phis[1]), _ld)
	
	# Create all directories for files to be saved.
	save_dir_names = ["images", "graphs", "models", "jsons", "videos"]
	for save_dir_name in save_dir_names:
		if os.path.exists(save_dir_name) == False:
			os.mkdir(save_dir_name)

	images_path = "images/{0}".format(test_name)

	if os.path.exists(images_path) == False:
		os.mkdir(images_path)

	# Initialize means and covariances.
	dimensionality = _dimensionality

	# Set the number of Gaussians
	gaussians = _gaussians

	# Set parameters to be trained.
	alphas = tf.Variable(_alphas, dtype=tf.float32, trainable=True, name="alphas")
	means = tf.Variable(_means, dtype=tf.float32, trainable=True, name="means")
	phis = tf.Variable(_phis, dtype=tf.float32, trainable=True, name="phis")
	covs = tf.Variable(_covs, dtype=tf.float32, trainable=True, name="covs")

	# Set hyper-parameters
	# We can select "Normal" and "Augmented"
	optim_method = "Augmented"
	max_iteration = 15000
	tot = 1e-5
	# lambda
	ld = tf.Variable(_ld, dtype=tf.float32, trainable=False, name="ld")
	# learning rate
	lr = tf.Variable(0.01, dtype=tf.float32, trainable=False, name="lr")

	if optim_method == "Augmented":
		mu = tf.Variable(1, dtype=tf.float32, trainable=False)
		update_step = 1000

	# Calculate normalized gaussians
	G = []
	for i in range(gaussians):
		G.append(unnormalized_gaussians(obs, means[i], covs[i], dimensionality))

	G = tf.convert_to_tensor(G, dtype=tf.float32, name="G")

	P = quantum_gmm(obs, G, alphas, gaussians, phis)
	P = tf.convert_to_tensor(P, dtype=tf.float32, name="P")

	Q = get_Q(P, gaussians); Q = tf.stop_gradient(Q)

	# Objective function, J
	J = None

	def loglikeihood(Q, P, gaussians):
		log_likelihood = 0
		for k in range(gaussians):
			log_likelihood += Q[k] * tf.math.log(tf.clip_by_value(P[k], 1e-10, 1))
		return tf.reduce_sum(log_likelihood, name="ll")
	
	def approx_constraint(G, alphas, phis, gaussians):
		mix_sum = 0
		for k in range(gaussians):
			for l in range(gaussians):
				if k == l:
					continue
				mix_sum += alphas[k] * alphas[l] * G[k] * G[l] * get_cosine(phis, k, l)
		return tf.math.abs(tf.reduce_sum(alphas ** 2) 
				+ tf.reduce_sum(mix_sum) - 1, name="constraint")

	# Normal Lagrangian multiplier
	if optim_method == "Normal":
		J = tf.add(-loglikeihood(Q, P, gaussians), 
			ld * approx_constraint(G, alphas, phis, gaussians), name="J")

	elif optim_method == "Augmented":
		# Augmented Lagrangian multiplier
		nll = -loglikeihood(Q, P, gaussians)
		constraint = approx_constraint(G, alphas, phis, gaussians)
		
		J = tf.add( tf.add(nll, -ld * constraint), 
				(1/(2*mu)) * (constraint ** 2),
				name="J")

	# Set optimizer to Adam with learning rate 0.01
	optim = tf.train.AdamOptimizer(learning_rate=lr)
	training = optim.minimize(J)

	# Build session
	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()

	sess.run(init)

	# Draw the first image.
	plot_clustered_data(dataset,
											means.eval(),
											covs.eval(),
											test_name,
											0,
											gaussians)

	# For graph
	saver = tf.train.Saver(max_to_keep=10000)
	saver.save(sess, "models/{0}/{1}.ckpt".format(test_name, 0))
	cur_J = sess.run(J)
	pre_J = cur_J

	# Train QGMM
	for i in range(1, max_iteration):
		sess.run(training)
		cur_J = sess.run(J)
		
		if abs(pre_J - cur_J) < tot:
			break

		if i % 100 == 0:
			print(i, test_name, cur_J, phis.eval())
			saver.save(sess, "models/{0}/{1}.ckpt".format(test_name, i),
					write_meta_graph=False)
			
			plot_clustered_data(dataset, 
													means.eval(),
													covs.eval(),
													test_name,
													i,
													gaussians)

		if optim_method == "Augmented" and i % update_step == 0:
			print("Lambda: ", ld.eval(), " mu: ", mu.eval())
			c = approx_constraint(G, alphas, phis, gaussians)
			if 0.1 <= c.eval():
				new_ld = tf.add(ld, -tf.div(c, mu))
				new_ld = tf.clip_by_value(new_ld, -1e7, 0)
				op = ld.assign(new_ld)
				sess.run(op)
				op = mu.assign(tf.clip_by_value(mu * 0.7, 1e-7, 1))
				sess.run(op)
		
		# Save the previous value of the objective function.
		pre_J = cur_J

	# After training finished, save the last state.
	saver.save(sess, "models/{0}/{1}.ckpt".format(test_name, i),
			write_meta_graph=False)
	plot_clustered_data(dataset, 
										means.eval(),
										covs.eval(),
										test_name,
										i,
										gaussians)
	sess.close()

if __name__== "__main__":
	dir_name = sys.argv[1]
	json_data = open("jsons/"+dir_name+"/"+dir_name+".json").read()

	data = json.loads(json_data)
	print(data)

	train_qgmm(_means=data["means"],
							_covs=data["covs"],
							_alphas=data["alphas"],
							_phis=data["phis"],
							_ld=data["ld"],
							_data=data["data"],
							_gaussians=data["gaussians"],
							_dimensionality=data["dimensionality"],
							_test_name=data["name"])
