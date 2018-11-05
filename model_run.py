import scipy.io as sio
import tensorflow as tf
import os

from data_helper import DataHelper
from model import DeepRthModel

BATCH_SIZE = 512
FRAMELEN = 10
OVERLAP = 5
EPOCH = 200

data = sio.loadmat("./datasets/pamap.mat")["data"]
dh = DataHelper(data, FRAMELEN, OVERLAP, [1, 0.1, 0.1])
ts_dim = data.shape[1]-1
model = DeepRthModel(r=10,
                     ts_dim=ts_dim,
                     timesteps=FRAMELEN,
                     encode_size=32,
                     cnn_filter_shapes=[[3, 3, 1, 16], [3, 3, 16, 32], [3, 3, 32, 64], [3, 3, 64, 64]],
                     cnn_strides=[[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
                     cnn_dense_layers=[256, 256],
                     rnn_hidden_states=256,
                     batch_size=BATCH_SIZE)
def model_train():
    """ """

    model.construct_loss()
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(model.loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(EPOCH):
            training_batch = dh.gen_training_batch(BATCH_SIZE)
            sample0 ,sample1, sample2 = training_batch
            feed_dict = {
                model.x0: sample0[0],
                model.corr0: sample0[1],
                model.x1: sample1[0],
                model.corr1: sample1[1],
                model.x2: sample2[0],
                model.corr2: sample2[1],
            }
            _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)
            print(loss)
        saver.save(sess, "checkpoints/model")

def model_test():
    """ """

    X, corrs, labels = dh.gen_test_samples()
    bencodes = model.binary_encode(X, corrs)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            model_train()
        test_encodes = sess.run(bencodes)

    sio.savemat("outputs/labels.mat", {"data": labels})
    sio.savemat("outputs/encodes.mat", {"data": test_encodes})

if __name__ == "__main__":
    model_train()
    model_test()
