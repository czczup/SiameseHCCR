import tensorflow as tf
from model import TripletNet
import numpy as np
import cv2
import sys
import heapq
import time
import os
import pandas as pd
import random


class Character:
    def __init__(self, label, feature=None):
        self.label = label
        self.feature = feature


def test(tripletNet, sess, dataset, train_time, debug=False, trainId=None):

    template_list = []
    for i, file in enumerate(os.listdir("database/anchor/")):
        if file.endswith("png"):
            image = cv2.imread("database/anchor/"+file, cv2.IMREAD_GRAYSCALE)
            image = image.reshape([64, 64, 1]) / 255.0
            feature = sess.run(tripletNet.positive_output, feed_dict={tripletNet.positive: [image], tripletNet.training: False})[0]
            template = Character(file[0], feature)
            template_list.append(template)
    template_feature_list = [template.feature for template in template_list]
    print("字符模板加载完成")
    print("字符模板总数为%d"%len(template_list))
    if not os.path.exists("file/"+trainId+"/results/train"):
        os.makedirs("file/"+trainId+"/results/train")
    if not os.path.exists("file/"+trainId+"/results/test"):
        os.makedirs("file/"+trainId+"/results/test")
    f = open("file/"+trainId+"/results/"+dataset+"/result%d.csv"%train_time, "w+", encoding='utf-8')
    range_len = range(len(template_list))

    path = "database/"+dataset
    top1, top5, top10, size = 0, 0, 0, 0
    for index, dir in enumerate(os.listdir(path)):
        # 载入图像
        filenames = []
        for count, file in enumerate(os.listdir(path+"/"+dir)):  # 加载文件夹中的所有图像
            filenames.append(path+"/"+dir+"/"+file)
        if dataset=='train':  # 训练集随机选择200个用于测试
            filenames = random.sample(filenames, 200)
            
        files = []
        for filename in filenames:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image = image.reshape([64, 64, 1]) / 255.0
            files.append(image)
            if debug: break

        # 提取图像特征
        features = sess.run(tripletNet.positive_output, feed_dict={tripletNet.positive: files, tripletNet.training: False})

        # 相似度网络计算样本对相似度
        predictions = []
        for feature in features:
            prediction = sess.run(tripletNet.test_y_hat, feed_dict={tripletNet.image_feature: [feature],
                                                                    tripletNet.template_feature: template_feature_list,
                                                                    tripletNet.training: False})
            predictions.append(prediction)


        # 找出相似度最大的11个汉字
        pred_characters = []
        for prediction in predictions:
            max_list = heapq.nlargest(11, range_len, prediction.take)
            pred_character = [template_list[item].label for item in max_list]
            pred_characters.append(pred_character)

        # 统计正确率
        count_top1, count_top5, count_top10 = 0, 0, 0
        length = len(pred_characters)
        count_dic = {}
        for item in pred_characters:
            if dir == item[0]:  # 统计top1正确率
                count_top1 += 1
                top1 += 1
            if dir in item[0:5]:  # 统计top5正确率
                count_top5 += 1
                top5 += 1
            if dir in item[0:10]:  # 统计top10正确率
                count_top10 += 1
                top10 += 1
            size += 1
            temp = list(item)
            if dir in temp:  # 输出top10错误汉字列表
                temp.remove(dir)
            for ch in temp[0:10]:  # 统计汉字的出错次数
                if ch in count_dic:
                    count_dic[ch] += 1
                else:
                    count_dic[ch] = 1
        else:
            ch_list = sorted(count_dic, key=lambda x: -count_dic[x])[:10]  # 找出出错次数最多的10个汉字
            if index == 0:
                print(count_dic)
            f.write(dir + "," + "".join(ch_list))
            f.write("," + str(count_top1 / length))
            f.write("," + str(count_top5 / length))
            f.write("," + str(count_top10 / length) + "\n")
        sys.stdout.write('\r>> Test image %d/%d' % (index + 1, 3755))
        sys.stdout.flush()
    else:
        f.write(str(top1/size))
        f.write(","+str(top5/size))
        f.write(","+str(top10/size)+"\n")
        f.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    with sess.graph.as_default():
        with sess.as_default():
            tripletNet = TripletNet()
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            var_list = [var for var in tf.global_variables() if "moving" in var.name]
            var_list += [var for var in tf.global_variables() if "global_step" in var.name]
            var_list += tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list)
            last_file = tf.train.latest_checkpoint("file/models/")
            print('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)
    test(tripletNet, sess, dataset="test")

