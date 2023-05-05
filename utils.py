import os
import io
import h5py
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import argparse

def load(path, **kwargs):
    ckpt = tf.train.Checkpoint(**kwargs)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              path,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    return ckpt_manager


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                   decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


class anneal():
    def __init__(self, init_val, final_val, delta=5):
        self.init_val = tf.constant(init_val, dtype="float32")
        self.final_val = tf.constant(final_val, dtype="float32")
        self.delta = tf.constant(delta, dtype="float32")
        self.lam = tf.Variable(0, dtype="float32")

    def __call__(self, progress):
        self.lam = ((2 / (1 + tf.exp(-self.delta * progress))) - 1)
        return ((1 - self.lam) * self.init_val) + (self.lam * self.final_val)


'''
Center Loss
'''


class CenterLoss():
    def __init__(self, batch_size, num_classes, len_features, alpha):
        self.centers = tf.Variable(tf.zeros([num_classes, len_features]),
                                   dtype=tf.float32,
                                   trainable=False)
        self.alpha = alpha
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.margin = tf.constant(100, dtype="float32")
        self.norm = lambda x: tf.reduce_sum(tf.square(x), 1)
        self.EdgeWeights = tf.ones((self.num_classes, self.num_classes)) - \
            tf.eye(self.num_classes)

    def get_center_loss(self, features, labels, alpha=None):
        if alpha is not None:
            self.alpha = alpha

        labels = tf.reshape(tf.argmax(labels, axis=-1), [-1])
        centers0 = tf.math.unsorted_segment_mean(features, labels,
                                                 self.num_classes)
        center_pairwise_dist = tf.transpose(self.norm(tf.expand_dims(centers0, 2) -
                                                      tf.transpose(centers0)))
        self.inter_loss = tf.math.reduce_sum(
            tf.multiply(tf.maximum(0.0, self.margin - center_pairwise_dist),
                        self.EdgeWeights))

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        centers_batch = tf.gather(self.centers, labels)
        diff = centers_batch - features
        diff /= tf.cast((1 + appear_times), tf.float32)
        diff *= self.alpha
        self.centers_update_op = tf.compat.v1.scatter_sub(
            self.centers, labels, diff)

        self.intra_loss = tf.nn.l2_loss(features - centers_batch)
        self.center_loss = self.intra_loss + self.inter_loss
        self.center_loss /= (self.num_classes * self.batch_size +
                             self.num_classes * self.num_classes)
        return self.center_loss


'''
MixUp: Regularization Strategy
https://arxiv.org/abs/1710.09412
args:
    x: 4D numpy array, with shape [batch_size, height, width, channels]
    y: 2D numpy array, with shape [batch_size, dim_logits]
    alpha: int, alpha parameter for beta distribution
output:
    data: numpy arrays, cutmix features and labels
'''


def mixup(x, y, alpha=1):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # random sample the lambda value from beta distribution.
    batch_size = x.get_shape().as_list()[0]
    weight = np.random.beta(alpha, alpha, batch_size)
    x_weight = weight.reshape(batch_size, 1, 1, 1)
    y_weight = weight.reshape(batch_size, 1)

    # Perform the mixup.
    indices = tf.random.shuffle(tf.range(batch_size))
    features = (x * x_weight) + (tf.gather(x, indices) * (1 - x_weight))
    labels = (y * y_weight) + (tf.gather(y, indices) * (1 - y_weight))

    return tf.stop_gradient(features), tf.stop_gradient(labels)


'''
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
Features https://arxiv.org/abs/1905.04899
args:
    x: 4D numpy array, with shape [batch_size, height, width, channels]
    y: 2D numpy array, with shape [batch_size, dim_logits]
    alpha: int, alpha parameter for beta distribution
output:
    data: numpy arrays, cutmix features and labels
'''


def cutmix(x, y, alpha=1):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    shape = x.get_shape()
    batch_size = shape[0]
    image_h = shape[1]
    image_w = shape[2]
    channels = shape[3]

    lam = np.random.beta(alpha, alpha)
    cx = tf.random.uniform(shape=[],
                           minval=0,
                           maxval=image_w,
                           dtype=tf.float32)
    cy = tf.random.uniform(shape=[],
                           minval=0,
                           maxval=image_h,
                           dtype=tf.float32)
    w = image_w * tf.sqrt(1 - lam)
    h = image_h * tf.sqrt(1 - lam)

    x0 = tf.cast(tf.round(tf.maximum(cx - w / 2, 0)), tf.int32)
    x1 = tf.cast(tf.round(tf.minimum(cx + w / 2, image_w)), tf.int32)
    y0 = tf.cast(tf.round(tf.maximum(cy - h / 2, 0)), tf.int32)
    y1 = tf.cast(tf.round(tf.minimum(cy + h / 2, image_h)), tf.int32)

    mask = tf.pad(
        tf.ones([batch_size, y1 - y0, x1 - x0, channels], dtype=tf.bool),
        [[0, 0], [y0, image_h - y1], [x0, image_w - x1], [0, 0]])
    indices = tf.random.shuffle(tf.range(batch_size))
    features = (x * tf.cast(tf.logical_not(mask), x.dtype)) + \
        (tf.gather(x, indices) * tf.cast(mask, x.dtype))
    labels = (y * lam) + (tf.gather(y, indices) * (1 - lam))

    return tf.stop_gradient(features), tf.stop_gradient(labels)


'''
Returns data, labels and classes from h5py file
args:
    filename: string, filename of h5py dataset
output:
    data: tuple, with (X_data, y_data, classes)
          where X_data and y_data are numpy arrays and classes is a list
'''


def get_h5dataset(filename):
    hf = h5py.File(filename, 'r')
    X_data = np.array(hf.get('X_data'))
    y_data = np.array(hf.get('y_data'))
    classes = list(hf.get('classes'))
    classes = [n.decode("ascii", "ignore") for n in classes]
    hf.close()
    return X_data, y_data, classes


'''
Balances the dataset to have same number of samples in every class and every day
args:
    X_data: numpy array, feature data [number_samples, ...]
    y_data: numpy array, label data [number_samples, 2], labels have to be sparse
                         with 1st dim for class label and 2nd dim for day
    num_days: int, total number of days in the dataset
    num_classes: int, total number of classes in the dataset
    max_samples_per_class: int, maximum number of samples to keep in each class per day
output:
    data: tuple, with (X_data, y_data, classes)
          where X_data and y_data are numpy arrays and classes is a list
'''


def balance_dataset(X_data,
                    y_data,
                    num_days=10,
                    num_classes=10,
                    max_samples_per_class=95):
    X_data_tmp, y_data_tmp = list(), list()
    for day in range(num_days):
        for idx in range(num_classes):
            X_data_tmp.extend(
                X_data[(y_data[:, 0] == idx)
                       & (y_data[:, 1] == day)][:max_samples_per_class])
            y_data_tmp.extend(
                y_data[(y_data[:, 0] == idx)
                       & (y_data[:, 1] == day)][:max_samples_per_class])
    return np.array(X_data_tmp), np.array(y_data_tmp)


def unbalance_dataset(X_data,
                      y_data,
                      min_data,
                      max_data,
                      num_days=10,
                      num_classes=10):
    step = (max_data - min_data) // (num_days-1)
    X_data_tmp, y_data_tmp = list(), list()
    for idx in range(num_classes):
        num_class = idx
        max_samples = min_data + (step * num_class)
        for day in range(num_days):
            query = (y_data[:, 0] == idx) & (y_data[:, 1] == day)
            X_data_tmp.extend(X_data[query][:max_samples])
            y_data_tmp.extend(y_data[query][:max_samples])
    return np.array(X_data_tmp), np.array(y_data_tmp)

def log_data(X_data,
             y_data,
             num_days=10,
             num_classes=10):
    for idx in range(num_classes):
        num_per_class_day = []
        for day in range(num_days):
            query = (y_data[:, 0] == idx) & (y_data[:, 1] == day)
            num_per_class_day.append(len(X_data[query]))
        print(idx, num_per_class_day)
'''
mean centers numpy array
args:
    X_data: numpy array, feature data [number_samples, ...]
    data_mean: None or double, mean value used to center data
               if None it is computed from X_data
output:
    data: tuple, with (X_data, data_mean)
          where X_data is a numpy arrays, data_mean is a double
'''


def mean_center(X_data, data_mean=None):
    if data_mean is None:
        data_mean = np.mean(X_data)
    X_data -= data_mean
    return X_data, data_mean


'''
normalizes numpy array to [-1, 1]
args:
    X_data: numpy array, feature data [number_samples, ...]
    data_min: None or double, minimum value used for normalization
              if None it is computed from X_data
    data_ptp: None or double, ptp value used for normalization
              if None it is computed from X_data
output:
    data: tuple, with (X_data, data_min, data_ptp)
          where X_data is a numpy arrays, data_min and data_ptp
          are doubles
'''


def normalize(X_data, data_min=None, data_ptp=None):
    if (data_ptp is None) or (data_min is None):
        data_min = np.min(X_data)
        data_ptp = np.ptp(X_data)
    X_data = 2. * (X_data - data_min) / data_ptp - 1
    return X_data, data_min, data_ptp


'''
preprocess target domain data
args:
    filename: string, filename of h5py dataset
    src_classes: list, class names from source domain
    train_trg_days: number of days to use as training data
output:
    X_train_trg: processed training features
    y_train_trg: processed training labels
    X_test_trg: processed testing features
    y_test_trg: processed testing labels
'''


def get_trg_data(filename, src_classes, train_trg_days, test_all=False, trgt_max=None):
    X_data_trg, y_data_trg, trg_classes = get_h5dataset(filename)

    # split days of data to train and test
    X_train_trg = X_data_trg[y_data_trg[:, 1] < train_trg_days]
    y_train_trg = y_data_trg[y_data_trg[:, 1] < train_trg_days]
    if trgt_max is not None and len(y_train_trg) > 0:
        trgt_max = [int(i) for i in trgt_max]
        X_train_trg, y_train_trg = unbalance_dataset(X_train_trg, y_train_trg, trgt_max[0], trgt_max[1])
    y_train_trg = y_train_trg[:, 0]
    y_train_trg = np.array([
        src_classes.index(trg_classes[y_train_trg[i]])
        for i in range(y_train_trg.shape[0])
    ])

    test_days = 0 if test_all else 3
    X_test_trg = X_data_trg[y_data_trg[:, 1] >= test_days]
    y_test_trg = y_data_trg[y_data_trg[:, 1] >= test_days, 0]
    y_test_trg = np.array([
        src_classes.index(trg_classes[y_test_trg[i]])
        for i in range(y_test_trg.shape[0])
    ])

    if (X_train_trg.shape[0] != 0):
        X_train_trg, trg_mean = mean_center(X_train_trg)
        X_train_trg, trg_min, trg_ptp = normalize(X_train_trg)
        y_train_trg = np.eye(len(src_classes))[y_train_trg]

        X_test_trg, _ = mean_center(X_test_trg, trg_mean)
        X_test_trg, _, _ = normalize(X_test_trg, trg_min, trg_ptp)
        y_test_trg = np.eye(len(src_classes))[y_test_trg]
    else:
        X_test_trg, _ = mean_center(X_test_trg)
        X_test_trg, _, _ = normalize(X_test_trg)
        y_test_trg = np.eye(len(src_classes))[y_test_trg]

    X_train_trg = X_train_trg.astype(np.float32)
    y_train_trg = y_train_trg.astype(np.uint8)
    X_test_trg = X_test_trg.astype(np.float32)
    y_test_trg = y_test_trg.astype(np.uint8)

    return X_train_trg, y_train_trg, X_test_trg, y_test_trg

def drop_with_noise(image, _min, _max):
    p = np.random.uniform(0, 1)
    if p<1/3:
        w = np.random.randint(1, image.shape[1])
        h = np.random.randint(2, 8)
    elif p>=1/3 and p<2/3:
        w = np.random.randint(2, 8)
        h = np.random.randint(1, image.shape[0])
    else:
        return
    left = np.random.randint(0, image.shape[1]-w)
    top = np.random.randint(0, image.shape[0]-h)
    image[top: top+h, left: left+w, :] = np.random.uniform(_min, _max, (h, w, 1))
def overlay_noise(image, _min, _max):
    p = np.random.uniform(0, 1)
    _length = (_max - _min)/10
    if p<2/3:
        image[0:image.shape[0], 0:image.shape[1], :] += np.random.uniform(-_length, _length, (image.shape[0], image.shape[1], 1))
def random_range(image):
    p = np.random.uniform(0, 1)
    if p<1/3:
        return
    _min = np.min(image)
    _max = np.max(image)
    _range = _max - _min
    _gap = _range/10
    n_min = np.random.uniform(_min-_gap, _min+_gap)
    n_max = np.random.uniform(_max-_gap, _max+_gap)
    n_range = n_max - n_min
    image[0:image.shape[0], 0:image.shape[1], :] = (image-_min) * n_range/_range + n_min

def preprocessing_function(image):
    np.random.seed(seed=None)
    _max = 0.9
    _min = -0.7
    # random_range(image) # tested but no difference
    drop_with_noise(image, _min, _max)
    drop_with_noise(image, _min, _max)
    overlay_noise(image, _min, _max)
    return image

class ImgGenDataset:
    def __init__(self, imgen, x_data, y_data, batch_size):
        self.imgen, self.x_data, self.y_data, self.batch_size = imgen, x_data, y_data, batch_size
        self.imgen_iter = iter(self.imgen.flow(self.x_data, self.y_data, batch_size=self.batch_size))
    def __len__(self):
        return self.x_data.shape[0]//self.batch_size # math.floor => drop_remainder
    def __iter__(self):
        return self
    def __next__(self):
        data = next(self.imgen_iter)
        if data[0].shape[0] != self.batch_size:
            data = next(self.imgen_iter)
        return data

class ImgGenAnchorDataset:
    def __init__(self, imgen_weak, imgen_strong, x_data, y_data, batch_size, u):
        self.seed=random.randrange(100)
        self.imgen_weak, self.imgen_strong, self.x_data, self.y_data, self.batch_size = imgen_weak, imgen_strong, x_data, y_data, batch_size
        self.batch_size_u = u * batch_size
        self.imgen_iter_weak = iter(self.imgen_weak.flow(self.x_data, self.y_data, batch_size=self.batch_size_u, seed=self.seed))
        self.imgen_iter_strong = iter(self.imgen_strong.flow(self.x_data, batch_size=self.batch_size_u, seed=self.seed))
    def __len__(self):
        return self.x_data.shape[0]//self.batch_size # math.floor => drop_remainder
    def __iter__(self):
        return self
    def __next__(self):
        x_weak, y = next(self.imgen_iter_weak)
        x_strong = next(self.imgen_iter_strong)
        if x_weak.shape[0] != self.batch_size_u:
            x_weak, y = next(self.imgen_iter_weak)
        if x_strong.shape[0] != self.batch_size_u:
            x_strong = next(self.imgen_iter_strong)
        return (x_weak, x_strong), y

class ImgGenAnchorMultiHardDataset:
    def __init__(self, imgen_weak, imgen_strongs, x_data, y_data, batch_size, u):
        self.seed=random.randrange(100)
        self.imgen_weak, self.imgen_strongs, self.x_data, self.y_data, self.batch_size = imgen_weak, imgen_strongs, x_data, y_data, batch_size
        self.batch_size_u = u * batch_size
        self.imgen_iter_weak = iter(self.imgen_weak.flow(self.x_data, self.y_data, batch_size=self.batch_size_u, seed=self.seed))
        self.imgen_iter_strongs = []
        for i in range(len(imgen_strongs)):
            self.imgen_iter_strongs.append(
                iter(self.imgen_strongs[i].flow(self.x_data, batch_size=self.batch_size_u, seed=self.seed)))
    def __len__(self):
        return self.x_data.shape[0]//self.batch_size # math.floor => drop_remainder
    def __iter__(self):
        return self
    def __next__(self):
        x_weak, y = next(self.imgen_iter_weak)
        if x_weak.shape[0] != self.batch_size_u:
            x_weak, y = next(self.imgen_iter_weak)
        
        x_strongs = []
        for i in range(len(self.imgen_iter_strongs)):
            x_strong = next(self.imgen_iter_strongs[i])
            if x_strong.shape[0] != self.batch_size_u:
                x_strong = next(self.imgen_iter_strongs[i])
            x_strongs.append(x_strong)
        return (x_weak, x_strongs), y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_one_hot(logits, num_classes=10):
    _argmax = tf.math.argmax(logits, axis=1)
    return tf.one_hot(_argmax, num_classes, dtype=tf.uint8)

class ExponentialMovingAverage:
    def __init__(self, decay):
        self.decay=decay
        self.current_value = None
        
    def apply(self,value, mask=None):
        if self.current_value is None:
            self.current_value = tf.Variable(value, dtype="float32")
        else:
            inverse_mask = mask*-1+1
            current_value = self.decay*self.current_value + (1-self.decay)* \
                                (mask*value + inverse_mask*self.current_value) # some may not have update values
            self.current_value.assign(current_value)
        return self.current_value

def cosine_similarity(feature_A, feature_B):
    #######################################################
    ########### This is the equivalent term: ##############
    #######################################################
    # feature_A_norm = tf.norm(feature_A, axis=-1, keepdims=True)
    # feature_B_norm = tf.norm(feature_B, axis=-1, keepdims=True)
    # cosine = tf.keras.backend.batch_dot(feature_A, feature_B, axes=1) / \
    #     tf.keras.backend.batch_dot(feature_A_norm, feature_B_norm, axes=1)
    # cosine = tf.squeeze(cosine)
    # return cosine

    return tf.keras.backend.batch_dot(tf.nn.l2_normalize(feature_A, axis=-1), 
                                    tf.nn.l2_normalize(feature_B, axis=-1), axes=1)