from model import Siamese
import tensorflow as tf
import time
import os
import pandas as pd
import cv2
import numpy as np


def read_and_decode_train(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image1': tf.FixedLenFeature([], tf.string),
                                           'image2': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image1 = tf.decode_raw(features['image1'], tf.uint8)
    image1 = tf.reshape(image1, [32, 32, 1])
    image1 = tf.cast(image1, tf.float32) / 255.0

    image2 = tf.decode_raw(features['image2'], tf.uint8)
    image2 = tf.reshape(image2, [32, 32, 1])
    image2 = tf.cast(image2, tf.float32) / 255.0

    label = tf.cast(features['label'], tf.int64)  # throw label tensor
    label = tf.reshape(label, [1])
    return image1, image2, label


def load_training_set(train_time):
    with tf.name_scope('input_train'):
        image_train1, image_train2, label_train = read_and_decode_train("file/tfrecord/train%d.tfrecord"%train_time)
        image_batch_train1, image_batch_train2, label_batch_train = tf.train.shuffle_batch(
            [image_train1, image_train2, label_train], batch_size=512, capacity=4096, min_after_dequeue=2048
        )
    return image_batch_train1, image_batch_train2, label_batch_train


def read_mapping():
    table = pd.read_csv("database/gb2312_level1.csv")
    value = table.values
    ids = [item[4] for item in value]
    chars = [item[2].strip() for item in value]
    id2char = dict(zip(ids, chars))
    char2id = dict(zip(chars, ids))
    return id2char, char2id


def train(sess, saver, siamese, writer, train_time, debug=False):
    BATCH_SIZE = 512
    DATA_SUM = 3000000 if not debug else 10000
    EPOCH = 20 if not debug else 1

    step_ = sess.run(siamese.global_step)
    if train_time > 0:
        step_ = step_ - train_time * EPOCH * (DATA_SUM // BATCH_SIZE)

    epoch_start = step_ // (DATA_SUM // BATCH_SIZE)
    step_start = step_ % DATA_SUM // BATCH_SIZE

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image_batch_train1, image_batch_train2, label_batch_train = load_training_set(train_time)

    for epoch in range(epoch_start, EPOCH):
        for step in range(step_start, DATA_SUM//BATCH_SIZE):
            time1 = time.time()
            image_train1, image_train2, label_train, step_ = sess.run(
                [image_batch_train1, image_batch_train2, label_batch_train, siamese.global_step])
            _, loss_ = sess.run([siamese.optimizer, siamese.loss], feed_dict={siamese.left: image_train1,
                                                                                 siamese.right: image_train2,
                                                                                 siamese.label: label_train,
                                                                                 siamese.training: True})
            print('[train %d, epoch %d, step %d/%d]: loss %.6f' % (train_time, epoch, step,
                                                                   int(DATA_SUM)//BATCH_SIZE, loss_),
                  'time %.3fs' % (time.time() - time1))
            if step_ % 10 == 0:
                image_train1, image_train2, label_train = sess.run(
                    [image_batch_train1, image_batch_train2, label_batch_train])
                acc_train, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_train1,
                                                                                                 siamese.right: image_train2,
                                                                                                 siamese.label: label_train,
                                                                                                 siamese.training: True})
                writer.add_summary(summary, step_)
                print('[train %d, epoch %d, step %d/%d]: train acc %.3f' % (train_time, epoch, step,
                                                                            int(DATA_SUM)//BATCH_SIZE, acc_train),
                      'time %.3fs' % (time.time() - time1))

            if step_ % 500 == 0:
                print("Save the model Successfully")
                saver.save(sess, "file/models/model.ckpt", global_step=step_)
    else:
        print("Save the model Successfully")
        saver.save(sess, "file/models/model.ckpt", global_step=step_)
        f = open("file/results/log/train%d.log"%train_time, "w+")
        f.close()

    coord.request_stop()
    coord.join(threads)
