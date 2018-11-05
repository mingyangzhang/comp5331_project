import scipy.io as sio
import numpy as np
from sklearn.metrics import average_precision_score

bencodes = sio.loadmat("outputs/encodes.mat")["data"]
labels  = sio.loadmat("outputs/labels.mat")["data"]

n, _ = bencodes.shape
bencodes = bencodes > 0

MAP = 0
for i in range(n):
    y_true = np.concatenate([labels[0, :i], labels[0, i+1:]])
    y_true = y_true == labels[0, i]
    y_encode = np.concatenate([bencodes[:i, :], bencodes[i+1:, :]])
    y_score = np.count_nonzero(y_encode != np.repeat(bencodes[i:i+1, :], n-1, axis=0), axis=1)
    MAP += average_precision_score(y_true, y_score)

MAP = MAP/n
import pdb;pdb.set_trace()