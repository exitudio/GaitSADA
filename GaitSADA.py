from sklearn.metrics import confusion_matrix
import h5py
import yaml
import shutil
import inspect
import argparse
import numpy as np
import tensorflow as tf
from resnet import ResNet50
from resnet_amca import ResNetAMCA, AM_logits
from utils import *
import sys
import os
from tqdm import tqdm
repo_path = os.getenv('MMWAVE_PATH')
print('repo_path:', repo_path)
sys.path.append(os.path.join(repo_path, 'models'))


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--epochs_2stage', type=int, default=10000)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--num_features', type=int, default=128)
    parser.add_argument('--model_filters', type=int, default=64)
    parser.add_argument('--activation_fn', default='selu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--train_src_days', type=int, default=3)
    parser.add_argument('--train_trg_days', type=int, default=0)
    parser.add_argument('--train_ser_days', type=int, default=0)
    parser.add_argument('--train_con_days', type=int, default=0)
    parser.add_argument('--train_off_days', type=int, default=0)
    parser.add_argument('--val', type=str2bool, nargs='?', default=False)
    parser.add_argument('--src_aug', type=int, default=0)
    parser.add_argument('--trgt_aug', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=.97)
    parser.add_argument('--checkpoint_path', default="checkpoints")
    parser.add_argument('--anneal', type=int, default=4)
    parser.add_argument('--trgt_max', nargs='+')
    parser.add_argument('--s', type=int, default=10)
    parser.add_argument('--m', type=float, default=0.2)
    parser.add_argument('--ca', type=float, default=1e-3)
    parser.add_argument('--dm_lambda', type=float, default=0.1)
    parser.add_argument('--log_dir', default="logs/example/GaitSADA/")
    parser.add_argument('--notes', default="")
    parser.add_argument('--notes_2stage', default="")
    return parser


def save_arg(arg):
    arg_dict = vars(arg)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    with open(os.path.join(arg.log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)


def get_cross_entropy_loss(labels, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    return tf.reduce_mean(loss)


@tf.function
def test_step(images):
    logits = model(images, training=False)
    return tf.nn.softmax(logits)

class ResNetAMCADomClas(ResNetAMCA):
    def __init__(self,
                num_classes,
                num_features,
                num_filters=64,
                activation='relu',
                regularizer='batchnorm',
                dropout_rate=0):
        super().__init__(num_classes, num_features, num_filters, activation,
                        regularizer, dropout_rate)
        
        self.emaCentroids = ExponentialMovingAverage(0.99)

    def call(self, x, training=False, output="logits"):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.max_pool1(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.avg_pool(x)
        fc1 = self.fc1(x)
        if output is "feature":
            return fc1
        logits = self.logits(fc1)
        if output is "align_centroid" or output is "align_centroid_always":
            centroids = self.get_feature_centroids(fc1, logits, output=="align_centroid")
            cosinesim = tf.matmul(tf.nn.l2_normalize(centroids, -1),
                    tf.nn.l2_normalize(self.logits.kernel, 0))
            centroid_sim_logits = AM_logits(labels=cls_labels, logits=cosinesim, m=m, s=s)
            return logits, centroid_sim_logits
        return logits

    def get_feature_centroids(self, feature, logits, is_pseudo):
        # centroid alignment
        vetor_size = feature.shape[1]
        if is_pseudo:
            softmax_feature = tf.stop_gradient(tf.nn.softmax(logits, -1))
            mask_confidence = tf.reduce_max(softmax_feature, axis=1) >= arg.confidence
        else:
            mask_confidence = tf.ones((feature.shape[0])) > 0
        # avoid the case that there is no feature > threshold
        if feature.shape is None:
            class_centroids = tf.ones((self.num_classes, vetor_size))
            mask_centroid = tf.zeros((self.num_classes, vetor_size))
        else:
            cls_idx = tf.math.argmax(logits, axis=1)
            class_centroids = []
            mask_centroid = []
            for cls in range(self.num_classes):
                mask_confidence_by_class = tf.math.logical_and(mask_confidence, cls_idx==cls)
                centroid_by_class = tf.boolean_mask(feature, mask_confidence_by_class)
                if centroid_by_class.shape[0] is None: # if no data in this class, mask out all
                    class_centroids.append(tf.zeros(vetor_size))
                    mask_centroid.append(tf.zeros(vetor_size))
                else: # if there are some data in this class, take mean
                    class_centroids.append(
                        tf.math.reduce_sum(centroid_by_class, axis=0) / centroid_by_class.shape[0]
                    )
                    mask_centroid.append(tf.ones(vetor_size))
            class_centroids = tf.stack(class_centroids)
            mask_centroid = tf.stack(mask_centroid)
        return self.emaCentroids.apply(class_centroids, mask_centroid)


@tf.function
def train_step(src_data, trg_data, s, m):
    src_images, src_labels = src_data
    (trgt_images_weak, trgt_images_strong), trg_labels = trg_data
    with tf.GradientTape() as tape:
        # supervised
        src_logits = model(src_images, training=True)
        src_logits = AM_logits(
            labels=src_labels, logits=src_logits, m=m, s=s)
        batch_cross_entropy_loss = get_cross_entropy_loss(labels=src_labels,
                                                          logits=src_logits)

        # self supervised
        trgt_weak_feature = model(trgt_images_weak,
                            training=True, output = "feature")
        trgt_strong_feature = model(trgt_images_strong,
                            training=True, output = "feature")

        self_supervised_loss = cosine_similarity(trgt_weak_feature, trgt_strong_feature)
        self_supervised_loss = tf.reduce_mean(self_supervised_loss, axis=1)    

        total_loss = batch_cross_entropy_loss + self_supervised_loss
    gradients = tape.gradient(batch_cross_entropy_loss,
                              model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    source_train_acc(src_labels, tf.nn.softmax(src_logits))
    cross_entropy_loss(total_loss)


@tf.function 
def train_step_seconstage(src_data, trg_data):
    src_images, src_labels = src_data
    (trgt_images_weak, trgt_images_strong), trg_labels = trg_data

    with tf.GradientTape() as tape:
        src_logits, _ = model(src_images,
                            training=True,
                            output="align_centroid_always")
        trgt_weak_logits, trgt_weak_centroid_sim_logits = model(trgt_images_weak,
                                                        training=True,
                                                        output="align_centroid")
        trgt_strong_logits = model(trgt_images_strong,
                            training=True)

        # amca:
        src_logits = AM_logits(
            labels=src_labels, logits=src_logits, m=0, s=s)

        one_hot_psuedo_labels = tf.math.argmax(trgt_weak_logits, axis=1)
        one_hot_psuedo_labels = tf.one_hot(one_hot_psuedo_labels, 10)
        trgt_weak_logits = AM_logits(labels=one_hot_psuedo_labels, logits=trgt_weak_logits, m=0, s=s)
        trgt_strong_logits = AM_logits(labels=one_hot_psuedo_labels, logits=trgt_strong_logits, m=0, s=s)

        # 1. supervised loss
        batch_cross_entropy_loss = get_cross_entropy_loss(labels=src_labels,
                                                          logits=src_logits)

        # 2. seconstage loss
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(trgt_weak_logits))
        # soft label
        loss_xeu = tf.nn.sigmoid_cross_entropy_with_logits(labels=pseudo_labels,
                                                            logits=trgt_strong_logits)
        loss_xeu = tf.reduce_mean(loss_xeu, axis=1)    
        # hard label
        # loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
        #                                                         logits=trgt_strong_logits)
        pseudo_mask = tf.cast(tf.reduce_max(pseudo_labels, axis=1) >= arg.confidence, tf.float32)
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)

        # centroids
        loss_centroid = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_labels,
                                                                logits=trgt_weak_centroid_sim_logits)
        loss_centroid = tf.reduce_mean(loss_centroid)

        total_loss = batch_cross_entropy_loss + 0.05*loss_centroid + loss_xeu
        seconstage_correct_rate(tf.math.equal(to_one_hot(pseudo_labels, 10), trg_labels))

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    teacher_rate(pseudo_mask)


def gen_dataset(x_data, y_data):
    if arg.trgt_aug>0:
        train_datasets.append(ImgGenDataset(imgen, x_data, y_data, batch_size=batch_size))
    else:
        data_set = tf.data.Dataset.from_tensor_slices(
            (x_data, y_data))
        data_set = data_set.shuffle(x_data.shape[0])
        data_set = data_set.batch(batch_size, drop_remainder=True)
        data_set = data_set.prefetch(batch_size)
        train_datasets.append(data_set)

def gen_weak_strong(x_data, y_data):
    imgen_weak = tf.keras.preprocessing.image.ImageDataGenerator()
    imgen_strong = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=[.8, 1.2],
        shear_range=5,
        rotation_range=5,
        preprocessing_function=preprocessing_function,
    )
    return ImgGenAnchorDataset(imgen_weak, imgen_strong, x_data, y_data, batch_size=batch_size, u=1)


if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()

    dataset_path = os.path.join(repo_path, 'data')
    num_classes = arg.num_classes
    batch_size = arg.batch_size
    train_src_days = arg.train_src_days
    train_ser_days = arg.train_ser_days
    train_con_days = arg.train_con_days
    train_trg_days = arg.train_trg_days
    train_off_days = arg.train_off_days
    epochs = arg.epochs
    init_lr = arg.init_lr
    num_features = arg.num_features
    activation_fn = arg.activation_fn
    model_filters = arg.model_filters
    anneal = arg.anneal
    s = arg.s
    m = arg.m
    ca = arg.ca
    dm_lambda = arg.dm_lambda

    num_domains = 0
    if train_ser_days > 0:
        num_domains += 1
    if train_con_days > 0:
        num_domains += 1
    if train_src_days > 0:
        num_domains += 1
    if train_trg_days > 0:
        num_domains += 1
    if train_off_days > 0:
        num_domains += 1

    seconstage_params = {
        "confidence": arg.confidence,
        "epochs_2stage": arg.epochs_2stage,
        "notes_2stage": arg.notes_2stage
    }
    sorted(seconstage_params)
    seconstage_params = str(seconstage_params).replace(" ","").replace("'","").replace(",","-")[1:-1]

    run_params = dict(vars(arg))
    del run_params['num_classes']
    del run_params['s']
    del run_params['m']
    del run_params['anneal']
    del run_params['activation_fn']
    del run_params['confidence']
    del run_params['log_dir']
    del run_params['checkpoint_path']
    del run_params['init_lr']
    del run_params['num_features']
    del run_params['model_filters']
    del run_params['batch_size']
    del run_params['trgt_max']
    del run_params['epochs_2stage']
    del run_params['notes_2stage']
    sorted(run_params)

    run_params = str(run_params).replace(" ",
                                         "").replace("'",
                                                     "").replace(",",
                                                                 "-")[1:-1]
    log_dir = os.path.join(repo_path, arg.log_dir, run_params)
    arg.log_dir = log_dir



    summary_writer_path = os.path.join(log_dir, "tensorboard_logs")
    checkpoint_path = os.path.join(log_dir, arg.checkpoint_path)

    save_arg(arg)
    shutil.copy2(inspect.getfile(ResNetAMCA), arg.log_dir)
    shutil.copy2(inspect.getfile(ImgGenDataset), arg.log_dir)
    shutil.copy2(os.path.abspath(__file__), arg.log_dir)


    '''
    Data Preprocessing
    '''
    X_data, y_data, classes = get_h5dataset(
        os.path.join(dataset_path, 'source_data.h5'))
    X_data, y_data = balance_dataset(X_data,
                                        y_data,
                                        num_days=10,
                                        num_classes=len(classes),
                                        max_samples_per_class=95)

    # split days of data to train and test
    X_src = X_data[y_data[:, 1] < train_src_days]
    y_src = y_data[y_data[:, 1] < train_src_days, 0]
    y_src = np.eye(len(classes))[y_src]
    X_train_src, X_test_src, y_train_src, y_test_src = train_test_split(
        X_src, y_src, stratify=y_src, test_size=0.10, random_state=42)

    X_trg = X_data[y_data[:, 1] >= train_src_days]
    y_trg = y_data[y_data[:, 1] >= train_src_days]
    X_train_trg = X_trg[y_trg[:, 1] < train_src_days + train_trg_days]
    y_train_trg = y_trg[y_trg[:, 1] < train_src_days + train_trg_days, 0]
    y_train_trg = np.eye(len(classes))[y_train_trg]

    X_test_trg = X_data[y_data[:, 1] >= train_src_days + train_trg_days]
    y_test_trg = y_data[y_data[:, 1] >= train_src_days + train_trg_days, 0]
    y_test_trg = np.eye(len(classes))[y_test_trg]

    del X_src, y_src, X_trg, y_trg, X_data, y_data

    # mean center and normalize dataset
    X_train_src, src_mean = mean_center(X_train_src)
    X_train_src, src_min, src_ptp = normalize(X_train_src)

    X_test_src, _ = mean_center(X_test_src, src_mean)
    X_test_src, _, _ = normalize(X_test_src, src_min, src_ptp)

    if (X_train_trg.shape[0] != 0):
        X_train_trg, trg_mean = mean_center(X_train_trg)
        X_train_trg, trg_min, trg_ptp = normalize(X_train_trg)

        X_test_trg, _ = mean_center(X_test_trg, trg_mean)
        X_test_trg, _, _ = normalize(X_test_trg, trg_min, trg_ptp)
    else:
        X_test_trg, _ = mean_center(X_test_trg, src_mean)
        X_test_trg, _, _ = normalize(X_test_trg, src_min, src_ptp)

    X_train_src = X_train_src.astype(np.float32)
    y_train_src = y_train_src.astype(np.uint8)
    X_test_src = X_test_src.astype(np.float32)
    y_test_src = y_test_src.astype(np.uint8)
    X_train_trg = X_train_trg.astype(np.float32)
    y_train_trg = y_train_trg.astype(np.uint8)
    X_test_trg = X_test_trg.astype(np.float32)
    y_test_trg = y_test_trg.astype(np.uint8)

    X_train_conf, y_train_conf, X_test_conf, y_test_conf = get_trg_data(
        os.path.join(dataset_path, 'target_conf_data.h5'), classes,
        train_con_days, trgt_max=arg.trgt_max)
    X_train_server, y_train_server, X_test_server, y_test_server = get_trg_data(
        os.path.join(dataset_path, 'target_server_data.h5'), classes,
        train_ser_days, trgt_max=arg.trgt_max)
    X_train_office, y_train_office, X_data_office, y_data_office = get_trg_data(os.path.join(
        dataset_path, 'target_office_data.h5'), classes,
        train_off_days, trgt_max=arg.trgt_max)

    print("Final shapes: ")
    print(" Train Src:   ", X_train_src.shape, y_train_src.shape, "\n",
          "Test Src:    ", X_test_src.shape, y_test_src.shape, "\n",
          "Train Trg:   ", X_train_trg.shape, y_train_trg.shape, "\n",
          "Test Trg:    ", X_test_trg.shape, y_test_trg.shape)
    print(" Train Conf:  ", X_train_conf.shape, y_train_conf.shape, "\n",
          "Test Conf:   ", X_test_conf.shape, y_test_conf.shape, "\n",
          "Train Server:", X_train_server.shape, y_train_server.shape, "\n",
          "Test Server: ", X_test_server.shape, y_test_server.shape, "\n",
          "Test office: ", X_data_office.shape, y_data_office.shape)

    # get tf.data objects for each set
    # Test
    conf_test_set = tf.data.Dataset.from_tensor_slices(
        (X_test_conf, y_test_conf))
    conf_test_set = conf_test_set.batch(batch_size, drop_remainder=False)
    conf_test_set = conf_test_set.prefetch(batch_size)

    server_test_set = tf.data.Dataset.from_tensor_slices(
        (X_test_server, y_test_server))
    server_test_set = server_test_set.batch(batch_size, drop_remainder=False)
    server_test_set = server_test_set.prefetch(batch_size)

    office_test_set = tf.data.Dataset.from_tensor_slices(
        (X_data_office, y_data_office))
    office_test_set = office_test_set.batch(batch_size, drop_remainder=False)
    office_test_set = office_test_set.prefetch(batch_size)

    src_test_set = tf.data.Dataset.from_tensor_slices((X_test_src, y_test_src))
    src_test_set = src_test_set.batch(batch_size, drop_remainder=False)
    src_test_set = src_test_set.prefetch(batch_size)

    time_test_set = tf.data.Dataset.from_tensor_slices(
        (X_test_trg, y_test_trg))
    time_test_set = time_test_set.batch(batch_size, drop_remainder=False)
    time_test_set = time_test_set.prefetch(batch_size)

    # Train
    train_datasets = []

    if arg.src_aug > 0 or arg.trgt_aug > 0:
        imgen = tf.keras.preprocessing.image.ImageDataGenerator(
            zoom_range=[.8, 1.2],
            shear_range=5,
            rotation_range=5,
            preprocessing_function=preprocessing_function,
        )

    if arg.src_aug > 0:
        train_datasets.append(ImgGenDataset(imgen, X_train_src, y_train_src, batch_size=batch_size))
    else:
        src_train_set = tf.data.Dataset.from_tensor_slices(
            (X_train_src, y_train_src))
        src_train_set = src_train_set.shuffle(X_train_src.shape[0])
        src_train_set = src_train_set.batch(batch_size, drop_remainder=True)
        src_train_set = src_train_set.prefetch(batch_size)
        train_datasets.append(src_train_set)

    if train_trg_days > 0:
        trgt_data = (X_train_trg, y_train_trg)
    if train_ser_days > 0:
        trgt_data = (X_train_server, y_train_server)
    if train_con_days > 0:
        trgt_data = (X_train_conf, y_train_conf)
    if train_off_days > 0:
        trgt_data = (X_train_office, y_train_office)
       
    if arg.val:
        X_train_trg_splt, X_test_trg_splt, y_train_trg_splt, y_test_trg_splt = train_test_split(
            trgt_data[0], trgt_data[1], stratify=trgt_data[1], test_size=0.3333, random_state=42)
        trgt_data = (X_train_trg_splt, y_train_trg_splt)
    gen_dataset(*trgt_data)
    '''
    Tensorflow Model
    '''

    source_train_acc = tf.keras.metrics.CategoricalAccuracy()
    target_test_acc = tf.keras.metrics.CategoricalAccuracy()

    cross_entropy_loss = tf.keras.metrics.Mean()
    domain_loss = tf.keras.metrics.Mean()
    teacher_rate = tf.keras.metrics.Mean()
    seconstage_correct_rate = tf.keras.metrics.Mean()

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        init_lr,
        decay_steps=(X_train_src.shape[0] // batch_size) * 200,
        end_learning_rate=init_lr * 1e-2,
        cycle=True)
    model = ResNetAMCADomClas(num_classes,
                              num_features,
                              num_filters=model_filters,
                              activation=activation_fn)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              checkpoint_path,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    m_anneal = tf.Variable(0, dtype="float32")

    if arg.val:
        test_set = tf.data.Dataset.from_tensor_slices((X_test_trg_splt, y_test_trg_splt))
        test_set = test_set.batch(batch_size, drop_remainder=False)
        test_set = test_set.prefetch(batch_size)
        y_test = y_test_trg_splt
        name_trg_acc = "val"
    elif train_trg_days > 0:
        test_set = time_test_set
        y_test = y_test_trg
        name_trg_acc = "time test acc" + str(train_trg_days)
    elif train_ser_days > 0:
        test_set = server_test_set
        y_test = y_test_server
        name_trg_acc = "server test acc" + str(train_ser_days)
    elif train_con_days > 0:
        test_set = conf_test_set
        y_test = y_test_conf
        name_trg_acc = "conference test acc" + str(train_con_days)
    elif train_off_days > 0:
        test_set = office_test_set
        y_test = y_data_office
        name_trg_acc = "office test acc" + str(train_off_days)

    # weak strong dataset
    weak_strong_ds = gen_weak_strong(*trgt_data)
    seconstage_train_dataset = [train_datasets[0], weak_strong_ds]        
    batch_per_epoch = min(map(len, seconstage_train_dataset))
    print('___ckpt_manager.latest_checkpoint:', ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print('--- LOAD CHECKPOINT ---')
    else:
        summary_writer = tf.summary.create_file_writer(summary_writer_path)
        for epoch in tqdm(range(epochs)):
            m_anneal.assign(tf.minimum(m * (epoch / (epochs / anneal)), m))
            for datasets in zip(*seconstage_train_dataset, range(batch_per_epoch)):
                train_step(*datasets[:2], s, m_anneal)

            if epoch % 50 == 0 or epoch == epochs-1:
                pred_labels = []
                for data in test_set:
                    pred_labels.extend(test_step(data[0]))
                target_test_acc(pred_labels, y_test)

                with summary_writer.as_default():
                    tf.summary.scalar(name_trg_acc,
                                    target_test_acc.result(),
                                    step=epoch)
                    tf.summary.scalar("source_train_acc",
                                    source_train_acc.result(),
                                    step=epoch)

            target_test_acc.reset_states()
            source_train_acc.reset_states()
            
        ckpt_save_path = ckpt_manager.save()
        print('Saved checkpoint at {}'.format(ckpt_save_path))

    summary_writer_path = os.path.join(log_dir, seconstage_params+"/tensorboard_logs")
    summary_writer = tf.summary.create_file_writer(summary_writer_path)
    seconstage_path = os.path.join(log_dir, seconstage_params)
    shutil.copy2(os.path.abspath(__file__), seconstage_path)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    pretrain_model = ResNetAMCADomClas(num_classes,
                                    num_features,
                                    num_filters=model_filters,
                                    activation=activation_fn)
    load(checkpoint_path, model=pretrain_model)


    checkpoint_path2 = os.path.join(log_dir, seconstage_params+"/checkpoints")
    ckpt2 = tf.train.Checkpoint(model=model)
    ckpt_manager2 = tf.train.CheckpointManager(ckpt2,
                                              checkpoint_path2,
                                              max_to_keep=1)
    ckpt2.restore(ckpt_manager2.latest_checkpoint).expect_partial()

    cls_labels = tf.range(0, 10)
    for epoch in range(arg.epochs_2stage):
        epoch += epochs
        for datasets in zip(*seconstage_train_dataset, range(batch_per_epoch)):
            train_step_seconstage(*datasets[:2])

        if epoch % 50 == 0 or epoch == epochs-1:
            pred_labels = []
            for data in test_set:
                pred_labels.extend(test_step(data[0]))
            target_test_acc(pred_labels, y_test)

            with summary_writer.as_default():
                tf.summary.scalar(name_trg_acc,
                                  target_test_acc.result(),
                                  step=epoch)
                tf.summary.scalar("teacher_rate",
                                  teacher_rate.result(),
                                  step=epoch)
                tf.summary.scalar("seconstage_correct_rate",
                                  seconstage_correct_rate.result(),
                                  step=epoch)
        target_test_acc.reset_states()
        teacher_rate.reset_states()
        seconstage_correct_rate.reset_states()
    
    ckpt_save_path2 = ckpt_manager2.save()
    print('Saved final checkpoint at {}'.format(ckpt_save_path2))
