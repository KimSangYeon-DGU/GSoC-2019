import os, sys, json
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from draw_utils import *
from record_utils import *
import pandas as pd
import numpy as np

test_name = sys.argv[1]
json_data = open("jsons/"+test_name+"/"+test_name+".json").read()

data = json.loads(json_data)

df = pd.read_csv('data/{0}'.format(data["data"]), sep=',')
dataset = df.to_numpy()
dataset = np.transpose(dataset)
gaussians = data["gaussians"]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	ckpt_state = tf.train.get_checkpoint_state("models/"+test_name)
	ckpt_paths = ckpt_state.all_model_checkpoint_paths
	
	graph = tf.get_default_graph()
	
	saver = tf.train.import_meta_graph("models/"+test_name+"/0.ckpt.meta")
	x_records = []
	nll_records = []
	constraint_records = []
	alpha_records = []
	phi_records = []
	gauss_records = []
	prob_records = []
	j_records = []
	
	open_type='w'
	i = 0
	for ckpt_path in ckpt_paths:
		print(i, ckpt_path)
		saver.restore(sess, ckpt_path)
		image_num = ckpt_path.split(".")
		image_num = int(image_num[0].split("/")[-1])
		'''
		plot_clustered_data(dataset, 
			graph.get_tensor_by_name('means:0').eval(),
			graph.get_tensor_by_name('covs:0').eval(),
			test_name,
			image_num,
			gaussians)
		'''
		x_records.append(image_num)
		ll = graph.get_tensor_by_name('ll:0').eval()
		nll_records.append(-ll)
		constraint_records.append(graph.get_tensor_by_name('constraint:0').eval())
		alpha_records.append(graph.get_tensor_by_name('alphas:0').eval())
		phi_records.append(graph.get_tensor_by_name('phis:0').eval())
		gauss_records.append(tf.reduce_sum(graph.get_tensor_by_name('G:0'), axis=1).eval())
		prob_records.append(tf.reduce_sum(graph.get_tensor_by_name('P:0'), axis=1).eval())
		j_records.append(graph.get_tensor_by_name('J:0').eval())

		i+=1

	record_graph(x_records,
                 nll_records,
                 constraint_records,
                 alpha_records,
                 phi_records,
                 gauss_records,
                 prob_records,
                 j_records,
                 test_name,
								 gaussians,
                 open_type='w')

	generate_video(test_name)