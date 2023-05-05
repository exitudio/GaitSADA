from ADDA.core import AdaptTrainer
from ADDA.helpers import load
from ADDA.models import MyClassifier
from sklearn.metrics import confusion_matrix
import h5py
import yaml
import shutil
import inspect
import argparse
import numpy as np
import tensorflow as tf
from resnet import ResNet50
from utils import *
import sys
import os
from ADDA.core import evaluate
import argparse
repo_path = os.getenv('MMWAVE_PATH')
sys.path.append(os.path.join(repo_path, 'models'))

num_classes = 10
activation_fn = 'selu'
log_images_freq = 25
save_freq = 25
batch_size = 64
num_features = 128
model_filters = 32
base_log_dir = "logs/Baselines/paper/ADDA/"


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_epochs', type=int, default=200)
    parser.add_argument('--train_src_days', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=1e-3)

    parser.add_argument('--target_epochs', type=int, default=10000)
    parser.add_argument('--tgt_lr', type=float, default=0.000006)
    parser.add_argument('--disc_lr', type=float, default=0.00003)
    parser.add_argument('--train_trg_days', type=int, default=0)
    parser.add_argument('--train_ser_days', type=int, default=0)
    parser.add_argument('--train_con_days', type=int, default=0)
    parser.add_argument('--train_off_days', type=int, default=0)
    return parser


def save_arg(arg):
    if isinstance(arg, argparse.Namespace):
        arg_dict = vars(arg)
    else:
        arg_dict = arg
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)


def get_cross_entropy_loss(labels, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    return tf.reduce_mean(loss)


@tf.function
def test_step(images):
    features = model(images, training=False)
    logits = classifier(features, training=False)
    return tf.nn.softmax(logits)


# @tf.function
def train_step(src_images, src_labels):
    with tf.GradientTape() as tape:
        src_features = model(src_images, training=True)
        predictions = classifier(src_features, training=True)
        batch_cross_entropy_loss = get_cross_entropy_loss(labels=src_labels,
                                                          logits=predictions)

    trainable_variables = model.trainable_variables + classifier.trainable_variables
    gradients = tape.gradient(batch_cross_entropy_loss,
                              trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    source_train_acc(src_labels, tf.nn.softmax(predictions))
    cross_entropy_loss(batch_cross_entropy_loss)

def set_converter(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(x.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(batch_size)

def write_acc(test_set, y_test):
    test_acc = tf.keras.metrics.CategoricalAccuracy()
    pred_labels = []
    for data in test_set:
        pred_labels.extend(test_step(data[0]))
    acc = test_acc(pred_labels, y_test).numpy().item()
    print('acc=', acc)

    path_acc='./tools/acc_generator/logs'
    if not os.path.exists(path_acc):
        os.makedirs(path_acc)
    with open(os.path.join(path_acc, "ADDA"), 'a') as f:
        f.write("train_src_days="+str(train_src_days)+" train_trg_days="+str(train_trg_days)+" train_ser_days="+str(train_ser_days)+" train_con_days="+str(train_con_days)+" train_off_days="+str(train_off_days)+"\n") 
        f.write("  acc="+str(acc)+"\n") 

if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()

    dataset_path = os.path.join(repo_path, 'data')
    train_src_days = arg.train_src_days
    train_ser_days = arg.train_ser_days
    train_con_days = arg.train_con_days
    train_trg_days = arg.train_trg_days
    train_off_days = arg.train_off_days
    source_epochs = arg.source_epochs
    target_epochs = arg.target_epochs
    init_lr = arg.init_lr
    pretrain_arg = {
        "source_epochs": arg.source_epochs,
        "train_src_days": arg.train_src_days,
        "init_lr": arg.init_lr
    }
    sorted(pretrain_arg)
    pretrain_params = str(pretrain_arg).replace(" ",
                                         "").replace("'",
                                                     "").replace(",",
                                                                 "-")[1:-1]
    log_dir = os.path.join(repo_path, base_log_dir, pretrain_params)

    summary_writer_path = os.path.join(log_dir, "tensorboard_logs")
    checkpoint_path = os.path.join(log_dir, "checkpoints")


    
    save_arg(pretrain_arg)
    shutil.copy2(inspect.getfile(ResNet50), log_dir)
    shutil.copy2(os.path.abspath(__file__), log_dir)
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
        train_con_days)
    X_train_server, y_train_server, X_test_server, y_test_server = get_trg_data(
        os.path.join(dataset_path, 'target_server_data.h5'), classes,
        train_ser_days)
    X_train_office, y_train_office, X_test_office, y_test_office = get_trg_data(os.path.join(
        dataset_path, 'target_office_data.h5'), classes,
        train_off_days)

    print("Final shapes: ")
    print(" Train Src:   ", X_train_src.shape, y_train_src.shape, "\n",
          "Test Src:    ", X_test_src.shape, y_test_src.shape, "\n",
          "Train Trg:   ", X_train_trg.shape, y_train_trg.shape, "\n",
          "Test Trg:    ", X_test_trg.shape, y_test_trg.shape)
    print(" Train Conf:  ", X_train_conf.shape, y_train_conf.shape, "\n",
          "Test Conf:   ", X_test_conf.shape, y_test_conf.shape, "\n",
          "Train Server:", X_train_server.shape, y_train_server.shape, "\n",
          "Test Server: ", X_test_server.shape, y_test_server.shape, "\n",
          "Train Office:", X_train_office.shape, y_train_office.shape, "\n",
          "Test office: ", X_test_office.shape, y_test_office.shape)


    src_train_set = set_converter(X_train_src, y_train_src)
    src_test_set = set_converter(X_test_src, y_test_src)

    # Add set
    if train_trg_days > 0:
        train_target_set = set_converter( X_train_trg, y_train_trg)
        test_target_set = set_converter( X_test_trg, y_test_trg)
        name_trg_acc = "time test acc" + str(train_trg_days)
    elif train_ser_days > 0:
        train_target_set = set_converter( X_train_server, y_train_server)
        test_target_set = set_converter( X_test_server, y_test_server)
        name_trg_acc = "server test acc" + str(train_ser_days)
    elif train_con_days > 0:
        train_target_set = set_converter( X_train_conf, y_train_conf)
        test_target_set = set_converter( X_test_conf, y_test_conf)
        name_trg_acc = "conference test acc" + str(train_con_days)
    elif train_off_days > 0:
        train_target_set = set_converter( X_train_office, y_train_office)
        test_target_set = set_converter( X_test_office, y_test_office)
        name_trg_acc = "office test acc" + str(train_off_days)

    '''
    Tensorflow Model
    '''

    source_train_acc = tf.keras.metrics.CategoricalAccuracy()
    cross_entropy_loss = tf.keras.metrics.Mean()

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        init_lr,
        decay_steps=(X_train_src.shape[0] // batch_size) * 200,
        end_learning_rate=init_lr * 1e-2,
        cycle=True)
    model = ResNet50(num_classes,
                     num_features,
                     num_filters=model_filters,
                     activation=activation_fn)
    classifier = MyClassifier()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    summary_writer = tf.summary.create_file_writer(summary_writer_path)
    ckpt_manager = load(checkpoint_path, model=model,
                        classifier=classifier, optimizer=optimizer)

    print('checkpoint_path:', checkpoint_path)
    if ckpt_manager.latest_checkpoint:
        print('---')
    else:
        for epoch in range(source_epochs):
            for source_data in src_train_set:
                train_step(source_data[0], source_data[1])

            with summary_writer.as_default():
                tf.summary.scalar("source_train_acc",
                                  source_train_acc.result(),
                                  step=epoch)
                tf.summary.scalar("cross_entropy_loss",
                                  cross_entropy_loss.result(),
                                  step=epoch)

            if (epoch + 1) % save_freq == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saved checkpoint for epoch {} at {}'.format(
                    epoch + 1, ckpt_save_path))

            source_train_acc.reset_states()
            cross_entropy_loss.reset_states()

        if save_freq != 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saved final checkpoint at {}'.format(ckpt_save_path))

    print("---- After -----")
    print('Train source')
    evaluate(src_train_set, model, classifier)
    print('Test source')
    evaluate(src_test_set, model, classifier)
    print('Train target')
    evaluate(train_target_set, model, classifier)
    print('Test target')
    evaluate(test_target_set, model, classifier)

    model_tgt = ResNet50(num_classes,
                         num_features,
                         num_filters=model_filters,
                         activation=activation_fn)
    load(checkpoint_path, model=model_tgt,
         classifier=classifier, optimizer=optimizer)

    # create save
    run_params = dict(vars(arg))
    sorted(run_params)
    run_params = str(run_params).replace(" ",
                                         "").replace("'",
                                                     "").replace(",",
                                                                 "-")[1:-1]
    log_dir = os.path.join(repo_path, base_log_dir, run_params)
    adda_dir = os.path.join(log_dir, 'ADDA')
    if os.path.exists(adda_dir):
        shutil.rmtree(adda_dir)
    shutil.copytree('./models/ADDA', adda_dir)
    save_arg(run_params)
    shutil.copy2(inspect.getfile(ResNet50), log_dir)
    shutil.copy2(os.path.abspath(__file__), log_dir)
    summary_writer_path = os.path.join(log_dir, "tensorboard_logs")
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    summary_writer = tf.summary.create_file_writer(summary_writer_path)
    

    ckpt2 = tf.train.Checkpoint(model=model_tgt)
    ckpt_manager2 = tf.train.CheckpointManager(ckpt2,
                                              checkpoint_path,
                                              max_to_keep=1)
    ckpt2.restore(ckpt_manager2.latest_checkpoint).expect_partial()

    if ckpt_manager2.latest_checkpoint:
        print('--- stage 2 LOAD CHECKPOINT ---')
    else:
        trainer = AdaptTrainer(model, model_tgt, classifier, src_train_set,
                            train_target_set, test_target_set, arg, summary_writer, checkpoint_path, name_trg_acc)
        trainer.train()
        ckpt_save_path2 = ckpt_manager2.save()
        print('Saved ADDA checkpoint at {}'.format(ckpt_save_path2))
