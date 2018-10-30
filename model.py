import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

def conv_layer(x, scope, kernel_shape, stride):
    x = tf.cast(x, tf.float32)
    with tf.variable_scope(scope + "-conv") as scp:
        kernel = tf.Variable(tf.truncated_normal(kernel_shape,
                                                dtype=tf.float32,
                                                stddev=1e-1),
                             name=scope + '-kernel')

        conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[kernel_shape[-1]], dtype=tf.float32),
                            trainable=True,
                            name='-biases')

        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope)

    with tf.name_scope(scope + '-lrn') as scp:
        lrn = tf.nn.local_response_normalization(conv,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)
    return lrn

class Model(object):

    def __init__(self):
        # self._layers = layers
        # self._input_dim = input_dim
        # self._output_dim = output_dim
        # self.batch_gen = batch_gen
        # self.test_gen = test_gen
        # self.learning_rate = learning_rate
        # self.epoches = epoches
        pass

    def cnn_forward(self, corr):
        conv1 = conv_layer(corr, "c1", [3, 3, 1, 16], [1, 1, 1, 1])
        conv2 = conv_layer(conv1, "c2", [3, 3, 16, 32], [1, 2, 2, 1])
        conv3 = conv_layer(conv2, "c3", [3, 3, 32, 64], [1, 2, 2, 1])
        conv4 = conv_layer(conv3, "c4", [3, 3, 64, 64], [1, 1, 1, 1])
        flat = tf.reshape(conv4, [-1, 25*25*64])
        dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
        return dense2

    def lstm_forward(self, X):
        X = tf.cast(X, tf.float32)
        timesteps = X.shape[1]
        X = tf.unstack(X, timesteps, 1)
        lstm_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)
        outputs, _ = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        return outputs

    def test(self):
        corr = np.random.rand(16, 100, 100, 1)
        dense = self.cnn_forward(corr)
        x = np.random.rand(16, 100, 10)
        outputs = self.lstm_forward(x)
        output = outputs[-1]
        encode = tf.concat([dense, output], axis=1)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            encode = sess.run(encode)
        import pdb;pdb.set_trace()

if __name__ == "__main__":
    model = Model()
    model.test()
