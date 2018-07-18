import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import module
import util


num_cluster = 10
batch_size = 128
eps = 1e-10  # term added for numerical stability of log computations


# ------------------------------------build the computation graph------------------------------------------
image_pool_input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name='image_pool_input')
u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

# get similarity matrix
label_feat = module.mnistNetwork(image_pool_input, num_cluster, name='mnistNetwork', reuse=False)
label_feat_norm = tf.nn.l2_normalize(label_feat, dim=1)
sim_mat = tf.matmul(label_feat_norm, label_feat_norm, transpose_b=True)

pos_loc = tf.greater(sim_mat, u_thres, name='greater')
neg_loc = tf.less(sim_mat, l_thres, name='less')
# select_mask = tf.cast(tf.logical_or(pos_loc, neg_loc, name='mask'), dtype=tf.float32)
pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)

# get clusters
pred_label = tf.argmax(label_feat, axis=1)

# define losses and train op
pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)

loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy)
train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
# 	train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)


# -------------------------------------------prepared datasets----------------------------------------------
# read mnist data (1 channel)
# mnist_1 = tf.contrib.learn.datasets.load_dataset("mnist")
mnist = input_data.read_data_sets('MNIST-data')  # your mnist data should be stored at 'MNIST-data'
mnist_train = mnist.train.images
mnist_train = np.reshape(mnist_train, (-1, 28, 28, 1))  # reshape into 3-channel image
mnist_train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
mnist_test = mnist.test.images
mnist_test = np.reshape(mnist_test, (-1, 28, 28, 1))  # reshape into 3-channel image
mnist_test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# # read cifar data
# cifar_data = []
# cifar_label = []
# for i in range(1, 6):
# 	file_name = 'cifar-10-data/' + 'data_batch_' + str(i)
# 	with open(file_name, 'rb') as fo:
# 		cifar_dict = cPickle.load(fo)
# 	data = cifar_dict['data']
# 	label = cifar_dict['labels']
	
# 	data = data.astype('float32')/255
# 	data = np.reshape(data, (-1, 3, 32, 32))
# 	data = np.transpose(data, (0, 2, 3, 1))
# 	cifar_data.append(data)
# 	cifar_label.append(label)

# cifar_data = np.concatenate(cifar_data, axis=0)
# cifar_label = np.concatenate(cifar_label, axis=0)
# # print cifar_data.shape


# --------------------------------------------run the graph-------------------------------------------------
base_lr = 0.001
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	sess.run(tf.global_variables_initializer())

	lamda = 0
	epoch = 1
	u = 0.95
	l = 0.455
	while u > l:
		u = 0.95 - lamda
		l = 0.455 + 0.1*lamda
		for i in range(1, int(1001)):  # 1000 iterations is roughly 1 epoch
			data_samples, _ = util.get_mnist_batch(batch_size, mnist_train, mnist_train_labels)
			feed_dict={image_pool_input: data_samples,
					   u_thres: u,
					   l_thres: l,
					   lr: base_lr}
			train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
			if i % 20 == 0:
				print('training loss at iter %d is %f' % (i, train_loss))

		lamda += 1.1 * 0.009

		# run testing every epoch
		data_samples, data_labels = util.get_mnist_batch(512, mnist_test, mnist_test_labels)
		feed_dict={image_pool_input: data_samples}
		pred_cluster = sess.run(pred_label, feed_dict=feed_dict)

		acc = util.clustering_acc(data_labels, pred_cluster)
		nmi = util.NMI(data_labels, pred_cluster)
		ari = util.ARI(data_labels, pred_cluster)
		print('testing NMI, ARI, ACC at epoch %d is %f, %f, %f.' % (epoch, nmi, ari, acc))

		if epoch % 5 == 0:  # save model at every proch
			model_name = 'DAC_ep_' + str(epoch) + '.ckpt'
			save_path = saver.save(sess, 'DAC_models/' + model_name)
			print("Model saved in file: %s" % save_path)

		epoch += 1


