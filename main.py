from generate_train_tfrecord import generate_train_tfrecord
from reconstruct_train_tfrecord import reconstruct_train_tfrecord
from train import train
from test import test
import tensorflow as tf
from model import Siamese
import os

def init_model(trainId):
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
            last_file = tf.train.latest_checkpoint("file/"+trainId+"/models/")
            if last_file:
                print('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)
            writer = tf.summary.FileWriter("file/"+trainId+"/logs/train", sess.graph)
    return sess, saver, siamese, writer


def main(trainId):
    train_time = 0
    debug = False
    if not os.path.exists("file/"+trainId):
        os.mkdir("file/"+trainId)
    if not os.path.exists("file/"+trainId+"/tfrecord/train0.tfrecord"):
        generate_train_tfrecord(train_time, sample_sum=500000, trainId=trainId)  # 生成第0个tfrecord
    while True:  # 无限循环
        sess, saver, siamese, writer = init_model(trainId=trainId)  # 每轮训练完成后，重新初始化计算图
        if not os.path.exists("file/"+trainId+"/results/log/train%d.log"%train_time):
            train(sess, saver, siamese, writer, train_time, debug=debug, trainId=trainId)  # 训练一定批次
        if not os.path.exists("file/"+trainId+"/results/train/result%d.csv"%train_time):
            test(siamese, sess, dataset="train", train_time=train_time, debug=debug, trainId=trainId)  # 用训练集测试
        if not os.path.exists("file/"+trainId+"/results/test/result%d.csv"%train_time):
            test(siamese, sess, dataset="test", train_time=train_time, debug=debug, trainId=trainId)  # 用测试集测试
        if not os.path.exists("file/"+trainId+"/tfrecord/train%d.tfrecord"%(train_time+1)):
            reconstruct_train_tfrecord(train_time, sample_sum=500000, trainId=trainId)  # 重构训练集
        tf.reset_default_graph()  # 清空计算图
        train_time += 1


if __name__ == '__main__':
    deviceId = input("please input device id (0-3): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    trainId = input("please input train id: ")
    main(trainId=trainId)
