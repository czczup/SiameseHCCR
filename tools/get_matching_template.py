import tensorflow as tf
from model import Siamese
import numpy as np
import cv2
import sys
import heapq
import time
import os
import pandas as pd


class Character:
    def __init__(self, image, label, feature=None):
        self.image = image.reshape([32, 32, 1])/255.0
        self.label = label
        self.feature = feature


def get_template_vector(siamese, sess):
    path = "database/train"
    avg_features = []
    for index, dir in enumerate(os.listdir(path)):
        # 载入匹配模板
        files = []
        for file in os.listdir(path+"/"+dir):  # 加载文件夹中的所有图像
            image = cv2.imread(path+"/"+dir+"/"+file, cv2.IMREAD_GRAYSCALE)
            image = image.reshape([32, 32, 1]) / 255.0
            files.append(image)

        # 提取图像特征
        features = sess.run(siamese.left_output, feed_dict={siamese.left: files, siamese.training: False})
        feature = np.mean(features, axis=0)
        avg_features.append(feature)
        sys.stdout.write('\r>> Generate template vector %d/%d'%(index+1, 3755))
        sys.stdout.flush()
        if index == 1 :
            break

    np.save("file/numpy/vector.npy", avg_features)
    print(np.load("file/numpy/vector.npy"))



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    with sess.graph.as_default():
        with sess.as_default():
            siamese = Siamese()
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            var_list = [var for var in tf.global_variables() if "moving" in var.name]
            var_list += [var for var in tf.global_variables() if "global_step" in var.name]
            var_list += tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list)
            last_file = tf.train.latest_checkpoint("file/models/")
            print('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
    get_template_vector(siamese, sess)

