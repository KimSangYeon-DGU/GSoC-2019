import os
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp



mytrain = "models/t_dist_3_180_1500/0.ckpt"
if os.path.exists(mytrain+".meta"):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		#chkp.print_tensors_in_checkpoint_file(mytrain, tensor_name='', all_tensors=True)
		ckpt_state = tf.train.get_checkpoint_state("models/t_dist_3_180_1500")
		ckpt_paths = ckpt_state.all_model_checkpoint_paths
		i = 0
		
		graph = tf.get_default_graph()
		
		saver = tf.train.import_meta_graph(mytrain+".meta")
		for ckpt_path in ckpt_paths:
			
			saver.restore(sess, ckpt_path)
	
			means = graph.get_tensor_by_name('means:0')
			print(i, means.eval())
			i+=1
		

