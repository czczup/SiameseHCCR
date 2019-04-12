from model import TripletNet
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
                                           'positive': tf.FixedLenFeature([], tf.string),
                                           'anchor': tf.FixedLenFeature([], tf.string),
                                           'negative': tf.FixedLenFeature([], tf.string),
                                       })  # return image and label
    positive = tf.decode_raw(features['positive'], tf.uint8)
    positive = tf.reshape(positive, [64, 64, 1])
    positive = tf.cast(positive, tf.float32) / 255.0

    anchor = tf.decode_raw(features['anchor'], tf.uint8)
    anchor = tf.reshape(anchor, [64, 64, 1])
    anchor = tf.cast(anchor, tf.float32)/255.0

    negative = tf.decode_raw(features['negative'], tf.uint8)
    negative = tf.reshape(negative, [64, 64, 1])
    negative = tf.cast(negative, tf.float32) / 255.0

    return positive, anchor, negative


def load_training_set(train_time, trainId):
    with tf.name_scope('input_train'):
        positive, anchor, negative = read_and_decode_train("file/"+trainId+"/tfrecord/train%d.tfrecord"%train_time)
        positive_batch, anchor_batch, negative_batch = tf.train.shuffle_batch(
            [positive, anchor, negative], batch_size=256,
            capacity=25600, min_after_dequeue=20000, num_threads=4
        )
    return positive_batch, anchor_batch, negative_batch


def read_mapping():
    table = pd.read_csv("database/gb2312_level1.csv")
    value = table.values
    ids = [item[4] for item in value]
    chars = [item[2].strip() for item in value]
    id2char = dict(zip(ids, chars))
    char2id = dict(zip(chars, ids))
    return id2char, char2id


def train(sess, saver, tripletNet, writer, train_time, debug=False, trainId=None):
    BATCH_SIZE = tripletNet.batch_size
    DATA_SUM = 300000 if not debug else 10000
    EPOCH = 15 if not debug else 1

    step_ = sess.run(tripletNet.global_step)
    if train_time > 0:
        step_ = step_ - train_time * EPOCH * (DATA_SUM // BATCH_SIZE)

    epoch_start = step_ // (DATA_SUM // BATCH_SIZE)
    step_start = step_ % (DATA_SUM // BATCH_SIZE)

    positive_batch_, anchor_batch_, negative_batch_ = load_training_set(train_time, trainId)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print(epoch_start)
    for epoch in range(epoch_start, EPOCH):
        for step in range(step_start, DATA_SUM//BATCH_SIZE):
            time1 = time.time()
            positive_batch, anchor_batch, negative_batch, step_ = sess.run(
                [positive_batch_, anchor_batch_, negative_batch_, tripletNet.global_step])
            _, loss_ = sess.run([tripletNet.optimizer, tripletNet.loss], feed_dict={tripletNet.positive: positive_batch,
                                                                                 tripletNet.anchor: anchor_batch,
                                                                                 tripletNet.negative: negative_batch,
                                                                                 tripletNet.training: True})
            print('[train %d, epoch %d, step %d/%d]: loss %.6f' % (train_time, epoch, step,
                                                                   int(DATA_SUM)//BATCH_SIZE, loss_),
                  'time %.3fs' % (time.time() - time1))
            if step_ % 10 == 0:
                positive_batch, anchor_batch, negative_batch = sess.run(
                    [positive_batch_, anchor_batch_, negative_batch_])
                acc_train, summary = sess.run([tripletNet.accuracy, tripletNet.merged], feed_dict={
                    tripletNet.positive: positive_batch, tripletNet.anchor: anchor_batch,
                    tripletNet.negative: negative_batch, tripletNet.training: True})
                writer.add_summary(summary, step_)
                print('[train %d, epoch %d, step %d/%d]: train acc %.3f' % (train_time, epoch, step,
                                                                            int(DATA_SUM)//BATCH_SIZE, acc_train),
                      'time %.3fs' % (time.time() - time1))

            if step_ % 500 == 0:
                print("Save the model Successfully")
                saver.save(sess, "file/"+trainId+"/models/model.ckpt", global_step=step_)
        else:
            step_start = 0
    else:
        print("Save the model Successfully")
        saver.save(sess, "file/"+trainId+"/models/model.ckpt", global_step=step_)
        if not os.path.exists("file/"+trainId+"/results/log"):
            os.makedirs("file/"+trainId+"/results/log")
        f = open("file/"+trainId+"/results/log/train%d.log"%train_time, "w+")
        f.close()

    coord.request_stop()
    coord.join(threads)
