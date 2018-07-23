import numpy as np
import random
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment


def get_cifar_batch(batch_size, cifar_data, cifar_label):
	batch_index = random.sample(range(len(cifar_label)), batch_size)

	batch_data = np.empty([batch_size, 32, 32, 3], dtype=np.float32)
	batch_label = np.empty([batch_size], dtype=np.int32)
	for n, i in enumerate(batch_index):
		batch_data[n, ...] = cifar_data[i, ...]
		batch_label[n] = cifar_label[i]

	return batch_data, batch_label


def get_mnist_batch(batch_size, mnist_data, mnist_labels):
	batch_index = random.sample(range(len(mnist_labels)), batch_size)
	
	batch_data = np.empty([batch_size, 28, 28, 1], dtype=np.float32)
	batch_label = np.empty([batch_size], dtype=np.int32)
	for n, i in enumerate(batch_index):
		batch_data[n, ...] = mnist_data[i, ...]
		batch_label[n] = mnist_labels[i]

	return batch_data, batch_label


def get_mnist_batch_test(batch_size, mnist_data, i):
	batch_data = np.copy(mnist_data[batch_size*i:batch_size*(i+1), ...])
	# batch_label = np.copy(mnist_labels[batch_size*i:batch_size*(i+1)])

	return batch_data


def get_svhn_batch(batch_size, svhn_data, svhn_labels):
	batch_index = random.sample(range(len(svhn_labels)), batch_size)

	batch_data = np.empty([batch_size, 32, 32, 3], dtype=np.float32)
	batch_label = np.empty([batch_size], dtype=np.int32)
	for n, i in enumerate(batch_index):
		batch_data[n, ...] = svhn_data[i, ...]
		batch_label[n] = svhn_labels[i]

	return batch_data, batch_label


def clustering_acc(y_true, y_pred):
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	ind = linear_assignment(w.max() - w)
	
	return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def NMI(y_true,y_pred):
	return metrics.normalized_mutual_info_score(y_true, y_pred)


def ARI(y_true,y_pred):
	return metrics.adjusted_rand_score(y_true, y_pred)
