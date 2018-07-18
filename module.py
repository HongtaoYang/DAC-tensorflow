import tensorflow as tf


def mnistNetwork(in_img, num_cluster, name='mnistNetwork', reuse=False):
	with tf.variable_scope(name, reuse=reuse):
		# conv1
		conv1 = tf.layers.conv2d(in_img, 64, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv1 = tf.layers.batch_normalization(conv1, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv1 = tf.nn.relu(conv1)
		# conv2
		conv2 = tf.layers.conv2d(conv1, 64, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv2 = tf.layers.batch_normalization(conv2, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv2 = tf.nn.relu(conv2)
		# conv3
		conv3 = tf.layers.conv2d(conv2, 64, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv3 = tf.nn.relu(conv3)
		conv3 = tf.layers.max_pooling2d(conv3, [2,2], [2,2])
		conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
		# conv4
		conv4 = tf.layers.conv2d(conv3, 128, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv4 = tf.layers.batch_normalization(conv4, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv4 = tf.nn.relu(conv4)
		# conv5
		conv5 = tf.layers.conv2d(conv4, 128, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv5 = tf.layers.batch_normalization(conv5, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv5 = tf.nn.relu(conv5)
		# conv6
		conv6 = tf.layers.conv2d(conv5, 128, [3,3], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv6 = tf.nn.relu(conv6)
		conv6 = tf.layers.max_pooling2d(conv6, [2,2], [2,2])
		conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
		# conv7
		conv7 = tf.layers.conv2d(conv6, 10, [1,1], [1,1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
		conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv7 = tf.nn.relu(conv7)
		conv7 = tf.layers.average_pooling2d(conv7, [2,2], [2,2])
		conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
		conv7_flat = tf.layers.flatten(conv7)

		# dense8
		fc8 = tf.layers.dense(conv7_flat, 10, kernel_initializer=tf.initializers.identity())
		fc8 = tf.layers.batch_normalization(fc8, axis=-1, epsilon=1e-5, training=True, trainable=False)
		fc8 = tf.nn.relu(fc8)
		# dense9
		fc9 = tf.layers.dense(fc8, num_cluster, kernel_initializer=tf.initializers.identity())
		fc9 = tf.layers.batch_normalization(fc9, axis=-1, epsilon=1e-5, training=True, trainable=False)
		fc9 = tf.nn.relu(fc9)

		out = tf.nn.softmax(fc9)

	return out