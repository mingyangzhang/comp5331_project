from .data_helper import DataHelper
from .model import DeepRthModel
BATCH_SIZE = 128
FRAMELEN = 10
OVERLAP = 5

def model_train():
    data = sio.loadmat("./datasets/pamap.mat")["data"]
    ts_dim = data.shape[1]
    dh = DataHelper(data, FRAMELEN, OVERLAP)
    model = DeepRthModel(ts_dim=ts_dim,
                         encode_size=32,
                         cnn_filter_shapes=[[3, 3, 1, 16], [3, 3, 16, 32], [3, 3, 32, 64], [3, 3, 64, 64]],
                         cnn_strides=[[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
                         cnn_dense_layers=[256, 128],
                         rnn_hidden_states=128,
                         batch_size=BATCH_SIZE)
    training_batch = dh.gen_training_batch(BATCH_SIZE)
