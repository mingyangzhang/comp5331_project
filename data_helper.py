import numpy as np
import scipy.io as sio

class DataHelper(object):
    """ Load data for model train and test. """
    def __init__(self, data, length, overlap, partition):
        """
            length: length of frame
            overlap: overlap of two frames
            partition: [train_size, validation_size, test_size]
        """

        labels = list(set(data[:, -1]))
        frames = {}
        for label in labels:
            frames[label] = []
        self._labels = labels
        self._frame = frames
        self._length = length
        self._overlap = overlap
        self._partition = partition
        col_mean = np.nanmean(data, axis=0)
        nan_inds = np.where(np.isnan(data))
        data[nan_inds] = np.take(col_mean, nan_inds[1])
        self._divide_into_frames(data)
        self._split_frames()

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

    def _split_frames(self):
        partition = []
        for i in range(len(self._partition)-1):
            partition.append(np.sum(self._partition[:i+1]))
        partition = partition/np.sum(self._partition)
        self._train_frames = dict.fromkeys(self._frame.keys())
        self._validation_frames = dict.fromkeys(self._frame.keys())
        self._test_frames = dict.fromkeys(self._frame.keys())
        for key in self._frame:
            size = self._frame[key].shape[0]
            splits = np.split(self._frame[key], [int(size*p) for p in partition])
            self._train_frames[key], self._validation_frames[key], self._test_frames[key] = splits

    def _gen_triples(self, batch_size):
        pos_index, neg_index = np.random.choice(len(self._labels), 2, replace=False)
        pos_label, neg_label = self._labels[pos_index], self._labels[neg_index]
        # select positive sample
        index = np.random.choice(self._train_frames[pos_label].shape[0], 2*batch_size, replace=False)
        pos_sample0 = self._train_frames[pos_label][index[:batch_size]]
        pos_sample1 = self._train_frames[pos_label][index[batch_size:]]

        # select negative sample
        index = np.random.choice(self._train_frames[neg_label].shape[0], batch_size, replace=False)
        neg_sample = self._train_frames[neg_label][index]

        return pos_sample0, pos_sample1, neg_sample

    def _compute_correlation_matrix(self, ts):
        ts = ts - np.mean(ts, axis=1, keepdims=True)
        divisor = np.sum(np.power(ts, 2), axis=0, keepdims=True)
        corr = np.divide(np.matmul(ts.T, ts), np.sqrt(np.matmul(divisor.T, divisor)))
        return corr

    def gen_training_batch(self, batch_size):
        samples = self._gen_triples(batch_size)
        _, _, ts_dim = samples[0].shape
        training_batch = []
        for sample in samples:
            corrs = np.zeros((batch_size, ts_dim, ts_dim, 1))
            for i in range(batch_size):
                corrs[i, :, :, 0] = self._compute_correlation_matrix(sample[i, :, :])
            training_batch.append((sample, corrs))
        return training_batch

    def gen_test_samples(self):
        test_frames = []
        test_labels = []
        for key in self._test_frames:
            size = self._test_frames[key].shape[0]
            test_frames.append(self._test_frames[key])
            test_labels.append(np.ones((size))*key)
        test_frames, test_labels = np.concatenate(test_frames), np.concatenate(test_labels)
        n, _, ts_dim = test_frames.shape
        corrs = np.zeros((n, ts_dim, ts_dim, 1))
        for i in range(n):
                corrs[i, :, :, 0] = self._compute_correlation_matrix(test_frames[i, :, :])
        return test_frames, corrs, test_labels

if __name__ == "__main__":
    data = sio.loadmat("./datasets/pamap.mat")["data"]
    dh = DataHelper(data, 10, 5)
    training_batch = dh.gen_training_batch(16)
    import pdb;pdb.set_trace()
