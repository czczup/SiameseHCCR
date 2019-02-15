import tensorflow as tf
import os
import random
import sys
import cv2
import pandas as pd


# 读取汉字对应关系
table = pd.read_csv("database/gb2312_level1.csv")
value = table.values
ids = [item[4] for item in value]
chars = [item[2].strip() for item in value]
id2char = dict(zip(ids, chars))
char2id = dict(zip(chars, ids))


def get_data():
    data = []
    path = "database/test/"
    length = len(os.listdir(path))
    for index, dir in enumerate(os.listdir(path)):
        for file in os.listdir(path+dir):
            label = char2id[dir]
            image_path = path + dir + "/" + file
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            data.append([image, label])
        sys.stdout.write('\r>> Loading test data %d/%d' % (index + 1, length))
        sys.stdout.flush()
    return data


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': int64_feature(class_id),
    }))


def _convert_dataset(data, tfrecord_path):
    """ Convert data to TFRecord format. """
    output_filename = os.path.join(tfrecord_path, "test.tfrecord")
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
    length = len(data)
    for index, item in enumerate(data):
        image = item[0].tobytes()
        label = item[1]
        example = image_to_tfexample(image, label)
        tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    data = get_data()
    random.seed(0)
    random.shuffle(data)
    _convert_dataset(data, "database/tfrecord/")
