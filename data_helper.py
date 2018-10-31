import numpy as np
import scipy.io as sio

class DataHelper(object):
    """ Load data for model train and test. """
    def __init__(self, data, length, overlap):
        labels = list(set(data[:, -1]))
        frames = {}
        for label in labels:
            frames[label] = []
        self._labels = labels
        self._frame = frames
        self._length = length
        self._overlap = overlap

        col_mean = np.nanmean(data, axis=0)
        nan_inds = np.where(np.isnan(data))
        data[nan_inds] = np.take(col_mean, nan_inds[1])
        self._divide_into_frames(data)

    def _divide_into_frames(self, data):
        n, _ = data.shape
        l, o = self._length, self._overlap
        step = l - o
        start = 0
        while(start+l<=n):
            frame = data[start:start+l, :-1]
            label = data[start, -1]
            if np.all(data[start:start+l, -1]==label):
                self._frame[label].append(frame)
            start += step
        for label in self._labels:
            self._frame[label] = np.array(self._frame[label])

    def _gen_triples(self, batch_size):
        pos_index, neg_index = np.random.choice(len(self._labels), 2, replace=False)
        pos_label, neg_label = self._labels[pos_index], self._labels[neg_index]
        # select positive sample
        index = np.random.choice(self._frame[pos_label].shape[0], 2*batch_size, replace=False)
        pos_sample0 = self._frame[pos_label][index[:batch_size]]
        pos_sample1 = self._frame[pos_label][index[batch_size:]]

        # select negative sample
        index = np.random.choice(self._frame[neg_label].shape[0], batch_size, replace=False)
        neg_sample = self._frame[neg_label][index]

        return pos_sample0, pos_sample1, neg_sample

    def _compute_correlation_matrix(self, ts):
        ts = ts - np.mean(ts, axis=1, keepdims=True)
        divisor = np.sum(np.power(ts, 2), axis=1, keepdims=True)
        corr = np.divide(np.matmul(ts, ts.T), np.sqrt(np.matmul(divisor, divisor.T)))
        return corr

    def gen_training_batch(self, batch_size):
        samples = self._gen_triples(batch_size)
        _, ts_dim, _ = samples[0].shape
        training_batch = []
        for sample in samples:
            corrs = np.zeros((batch_size, ts_dim, ts_dim))
            for i in range(batch_size):
                corrs[i, :, :] = self._compute_correlation_matrix(sample[i, :, :])
            training_batch.append((sample, corrs))
        return training_batch

    def gen_test_sample(self):
        pass

if __name__ == "__main__":
    data = sio.loadmat("./datasets/pamap.mat")["data"]
    dh = DataHelper(data, 10, 5)
    training_batch = dh.gen_training_batch(16)
    import pdb;pdb.set_trace()
