import tensorflow as tf
from functools import reduce
from operator import mul


class TripletNet(object):
    def __init__(self):
        with tf.name_scope("input"):
            self.positive = tf.placeholder(tf.float32, [None, 64, 64, 1], name='positive')
            self.anchor = tf.placeholder(tf.float32, [None, 64, 64, 1], name='anchor')
            self.negative = tf.placeholder(tf.float32, [None, 64, 64, 1], name='negative')

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        with tf.variable_scope("Triplet") as scope:
            self.positive_output = self.model(self.positive)
            scope.reuse_variables()
            self.negative_output = self.model(self.negative)
            scope.reuse_variables()
            self.anchor_output = self.model(self.anchor)

        self.prediction, self.loss, self.test_param = self.my_loss()
        tf.summary.scalar('loss', self.loss)
        self.batch_size = 256
        self.ones = tf.ones([self.batch_size, 1])
        self.zeros = tf.zeros([self.batch_size, 1])
        with tf.name_scope('correct_prediction'):
            correct_prediction1 = tf.equal(tf.less(self.prediction[0], 0.5), tf.less(self.ones, 0.5))
            correct_prediction2 = tf.equal(tf.less(self.prediction[1], 0.5), tf.less(self.zeros, 0.5))
            correct_prediction = tf.concat([correct_prediction1,correct_prediction2], axis=-1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(1E-3).minimize(self.loss, global_step=self.global_step)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('matching-accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()
        with tf.name_scope('test-network'):
            self.test_network()


    def conv2d(self, x, output_filters, kernel, strides=1, padding="SAME"):
        conv = tf.contrib.layers.conv2d(x, output_filters, [kernel, kernel], activation_fn=tf.nn.relu, padding=padding,
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        stride=strides)
        return conv

    def residual(self, x, num_filters, kernel, strides, with_shortcut=False):
        with tf.name_scope("residual"):
            conv1 = self.conv2d(x, num_filters[1], kernel=1, strides=strides)
            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            relu1 = tf.nn.relu(bn1)
            conv2 = self.conv2d(relu1, num_filters[2], kernel=3)
            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            relu2 = tf.nn.relu(bn2)
            conv3 = self.conv2d(relu2, num_filters[3], kernel=1)
            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            if with_shortcut:
                shortcut = self.conv2d(x, num_filters[3], kernel=1, strides=strides)
                bn_shortcut = tf.layers.batch_normalization(shortcut, training=self.training)
                residual = tf.nn.relu(bn_shortcut + bn3)
            else:
                residual = tf.nn.relu(x + bn3)
            return residual

    def model(self, x):
        channel = 32
        with tf.variable_scope("conv1") as scope:
            conv1 = self.conv2d(x, channel, 7, 1)
            bn = tf.layers.batch_normalization(conv1, training=self.training)
            relu = tf.nn.relu(bn)
            pool = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
        with tf.variable_scope("block1") as scope:
            res = self.residual(pool, [channel, channel//2, channel//2, channel*2], 3, 1, with_shortcut=True)
            # res = self.residual(res, [channel*2, channel//2, channel//2, channel*2], 3, 1)
            print(res)
        with tf.variable_scope("block2") as scope:
            res = self.residual(res, [channel*2, channel, channel, channel*4], 3, 2, with_shortcut=True)
            # res = self.residual(res, [channel*4, channel, channel, channel*4], 3, 1)
            print(res)
        with tf.variable_scope("block3") as scope:
            res = self.residual(res, [channel*4, channel*2, channel*2, channel*8], 3, 2, with_shortcut=True)
            # res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 3, 1)
            print(res)
        with tf.variable_scope("block4") as scope:
            res = self.residual(res, [channel*8, channel*4, channel*4, channel*16], 3, 2, with_shortcut=True)
            # res = self.residual(res, [channel*16, channel*4, channel*4, channel*16], 3, 1)
            print(res)
            pool = tf.nn.avg_pool(res, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            flatten = tf.layers.flatten(pool)  # 2*2*1024=4096
            print(flatten)
        with tf.variable_scope("fc1") as scope:
            dense = tf.layers.dense(flatten, units=256, activation=None)
        return dense

    def my_loss(self):
        W = tf.Variable(tf.random_normal([256, 1], stddev=0.1), name='W')
        b = tf.Variable(tf.zeros([1, 1])+0.1, name='b')
        with tf.variable_scope("positive-anchor"):
            difference = tf.abs(self.positive_output-self.anchor_output)
            wx_plus_b = tf.matmul(difference, W)+b
            y_hat1 = tf.nn.sigmoid(wx_plus_b)
        with tf.variable_scope("negative-anchor"):
            difference = tf.abs(self.negative_output-self.anchor_output)
            wx_plus_b = tf.matmul(difference, W)+b
            y_hat2 = tf.nn.sigmoid(wx_plus_b)

        with tf.name_scope("loss"):
            # log趋向于0，y_hat1趋向于1，y_hat2趋向于0
            losses = -(tf.log(y_hat1+1E-9) + tf.log(1-y_hat2+1E-9))
            # losses = tf.nn.l2_loss(y_hat1-1) + tf.nn.l2_loss(y_hat2)
            loss = tf.reduce_mean(losses)
        return [y_hat1, y_hat2], loss, [W, b]

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            print(variable)
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    def test_network(self):
        self.template_feature = tf.placeholder(tf.float32, [None, 256])
        self.image_feature = tf.placeholder(tf.float32, [None, 256])
        image_feature = tf.tile(self.image_feature, multiples=[3755, 1])
        output_difference = tf.abs(image_feature-self.template_feature)
        wx_plus_b = tf.matmul(output_difference, self.test_param[0])+self.test_param[1]
        self.test_y_hat = tf.nn.sigmoid(wx_plus_b, name='distance')

if __name__ == '__main__':
    model = TripletNet()
    var_list = tf.global_variables()
    for var in var_list:
        if "batch" not in var.name and "Adam" not in var.name:
            print(var)