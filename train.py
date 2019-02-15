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


def read_and_decode_test(filename):
    filename_queue = tf.train.string_input_producer([filename])  # create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [32, 32, 1])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(features['label'], tf.int64)  # throw label tensor
    # label = tf.reshape(label, [1])
    return image, label


def load_training_set():
    with tf.name_scope('input_train'):
        image_train1, image_train2, label_train = read_and_decode_train("database/tfrecord/train.tfrecord")
        image_batch_train1, image_batch_train2, label_batch_train = tf.train.shuffle_batch(
            [image_train1, image_train2, label_train], batch_size=512, capacity=10240, min_after_dequeue=5120
        )
    return image_batch_train1, image_batch_train2, label_batch_train


def load_test_set():
    with tf.name_scope('input_test'):
        image_test, label_test = read_and_decode_test("database/tfrecord/test.tfrecord")
        image_batch_test, label_batch_test = tf.train.shuffle_batch(
            [image_test, label_test], batch_size=256, capacity=2560, min_after_dequeue=1280
        )
    return image_batch_test, label_batch_test


def read_mapping():
    table = pd.read_csv("database/gb2312_level1.csv")
    value = table.values
    ids = [item[4] for item in value]
    chars = [item[2].strip() for item in value]
    id2char = dict(zip(ids, chars))
    char2id = dict(zip(chars, ids))
    return id2char, char2id


def train():
    batch_size = 512
    data_sum = 3E6
    siamese = Siamese()

    image_batch_train1, image_batch_train2, label_batch_train = load_training_set()
    image_batch_test, label_batch_test = load_test_set()

    # Adaptive use of GPU memory.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    id2char, char2id = read_mapping()

    with tf.Session(config=tf_config) as sess:
        # general setting
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # Recording training process.
        writer_train = tf.summary.FileWriter("file/logs/train", sess.graph)
        writer_test = tf.summary.FileWriter("file/logs/test", sess.graph)

        last_file = tf.train.latest_checkpoint("file/models")
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += [var for var in tf.global_variables() if "global_step" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
        if last_file:
            tf.logging.info('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
        # train
        while 1:
            time1 = time.time()
            image_train1, image_train2, label_train, step = sess.run(
                [image_batch_train1, image_batch_train2, label_batch_train, siamese.global_step])
            _, loss_ = sess.run([siamese.optimizer, siamese.loss], feed_dict={siamese.left: image_train1,
                                                                              siamese.right: image_train2,
                                                                              siamese.label: label_train,
                                                                              siamese.training: True})
            print('[epoch %d, step %d/%d]: loss %.6f' % (step // (data_sum // batch_size),
                                                         step % (data_sum // batch_size),
                                                         data_sum // batch_size, loss_),
                  'time %.3fs' % (time.time() - time1))
            if step % 10 == 0:
                image_train1, image_train2, label_train = sess.run(
                    [image_batch_train1, image_batch_train2, label_batch_train])
                acc_train, summary = sess.run([siamese.accuracy, siamese.merged], feed_dict={siamese.left: image_train1,
                                                                                             siamese.right: image_train2,
                                                                                             siamese.label: label_train,
                                                                                             siamese.training: True})
                writer_train.add_summary(summary, step)

                print('[epoch %d, step %d/%d]: train acc %.3f' % (step // (data_sum // batch_size),
                                                                  step % (data_sum // batch_size),
                                                                  data_sum // batch_size, acc_train),
                      'time %.3fs' % (time.time() - time1))
            # if step % 10 == 0:
            #     files = []
            #     time1 = time.time()
            #     for id in range(3755):
            #         filename = "database/train/"+id2char[id]+"/0.png"
            #         image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            #         image = np.reshape(image, [32, 32, 1])
            #         files.append(image)
            #     templates = sess.run(siamese.left_output, feed_dict={siamese.left: files, siamese.training: False})
            #     image_test, label_test = sess.run([image_batch_test, label_batch_test])
            #     top10, top5, top2, top1, y_hat, summary_test = sess.run([siamese.test_accuracy[0][0], siamese.test_accuracy[1][0],
            #                                                       siamese.test_accuracy[2][0], siamese.test_accuracy[3][0],
            #                                                              siamese.y_hat,
            #                                                       siamese.merged_test],
            #                                                       feed_dict={siamese.left: image_test,
            #                                                                  siamese.template_feature: templates,
            #                                                                  siamese.index: label_test,
            #                                                                  siamese.training: False})
            #     print("top1: %.4f, top2: %.4f, top5: %.4f, top10: %.4f," % (top1, top2, top5, top10),
            #           'time %.3fs' % (time.time() - time1))
            #     print(y_hat)
            #     print(len(y_hat))
            #     print(len(y_hat[0]))
            #
            #     writer_test.add_summary(summary_test, step)
            if step % 500 == 0:
                print("Save the model Successfully")
                saver.save(sess, "file/models/model.ckpt", global_step=step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    deviceID = input("please input the device ID: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceID
    train()
