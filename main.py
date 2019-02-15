from generate_train_tfrecord import generate_train_tfrecord
from reconstruct_train_tfrecord import reconstruct_train_tfrecord
from train import train
from test import test
import tensorflow as tf
from model import Siamese
import os

def init_model():
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
            saver = tf.train.Saver(var_list=var_list, max_to_keep=20)
            last_file = tf.train.latest_checkpoint("file/models/")
            if last_file:
                print('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)
            writer = tf.summary.FileWriter("file/logs/train", sess.graph)
    return sess, saver, siamese, writer

def reset_graph(sess):
    tf.reset_default_graph()
    with sess.graph.as_default():
        with sess.as_default():
            siamese = Siamese()
    return siamese

def main():
    train_time = 0
    debug = False
    generate_train_tfrecord(train_time, sample_sum=1500000)  # 生成第0个tfrecord
    while True:  # 无限循环
        sess, saver, siamese, writer = init_model()
        train(sess, saver, siamese, writer, train_time, debug=debug)  # 训练一定批次
        test(siamese, sess, dataset="train", train_time=train_time, debug=debug)  # 用训练集测试
        test(siamese, sess, dataset="test", train_time=train_time, debug=debug)  # 用测试集测试
        reconstruct_train_tfrecord(train_time, sample_sum=1500000)  # 重构训练集
        tf.reset_default_graph()
        train_time += 1


if __name__ == '__main__':
    main()
