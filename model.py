import tensorflow as tf
from functools import reduce
from operator import mul


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      if g != None:
        expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


class Siamese(object):
    def __init__(self):
        self.batch_size = 128
        self.gpu_num = 4

        with tf.name_scope("input"):
            self.left = tf.placeholder(tf.float32, [None, 32, 32, 1], name='left')
            self.right = tf.placeholder(tf.float32, [None, 32, 32, 1], name='right')
            label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
            self.label = tf.to_float(label)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(1e-3, self.global_step, decay_steps=1E6//(self.batch_size*self.gpu_num),
                                                        decay_rate=0.999, staircase=True)
        tf.summary.scalar('learning_rate', self.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(4):
                with tf.device("/gpu:%d"%i):
                    with tf.name_scope("tower_%d"%i):
                        _left = self.left[i*self.batch_size:(i+1)*self.batch_size]
                        _right = self.right[i*self.batch_size:(i+1)*self.batch_size]
                        _label = self.label[i*self.batch_size:(i+1)*self.batch_size]
                        with tf.variable_scope("Siamese") as scope:
                            left_output = self.model(_left)
                            scope.reuse_variables()
                            right_output = self.model(_right)
                            prediction, loss = self.contrastive_loss(left_output, right_output, _label)
                        tf.get_variable_scope().reuse_variables()
                        with tf.device("/cpu:0"):
                            tf.summary.scalar('loss_%d'%i, loss)
                        grads = self.optimizer.compute_gradients(loss)
                        tower_grads.append(grads)

                        if i==0:
                            self.loss = loss
                            self.left_output = left_output
                            self.right_output = right_output
                            with tf.name_scope('correct_prediction'):
                                correct_prediction = tf.equal(tf.less(prediction, 0.5), tf.less(_label, 0.5))

                            with tf.name_scope('accuracy'):
                                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                                with tf.device("/cpu:0"):
                                    tf.summary.scalar('matching-accuracy', self.accuracy)
                                    self.merged = tf.summary.merge_all()
                            #
                            with tf.name_scope('test-network'):
                                self.test_network()

        grads = average_gradients(tower_grads)
        self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

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
            res = self.residual(res, [channel*2, channel//2, channel//2, channel*2], 3, 1)
            res = self.residual(res, [channel*2, channel//2, channel//2, channel*2], 3, 1)
            print(res)
        with tf.variable_scope("block2") as scope:
            res = self.residual(res, [channel*2, channel, channel, channel*4], 3, 2, with_shortcut=True)
            res = self.residual(res, [channel*4, channel, channel, channel*4], 3, 1)
            res = self.residual(res, [channel*4, channel, channel, channel*4], 3, 1)
            print(res)
        with tf.variable_scope("block3") as scope:
            res = self.residual(res, [channel*4, channel*2, channel*2, channel*8], 3, 2, with_shortcut=True)
            res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 3, 1)
            res = self.residual(res, [channel*8, channel*2, channel*2, channel*8], 3, 1)
            print(res)
        with tf.variable_scope("block4") as scope:
            res = self.residual(res, [channel*8, channel*4, channel*4, channel*16], 3, 2, with_shortcut=True)
            res = self.residual(res, [channel*16, channel*4, channel*4, channel*16], 3, 1)
            res = self.residual(res, [channel*16, channel*4, channel*4, channel*16], 3, 1)
            print(res)
            pool = tf.nn.avg_pool(res, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
            flatten = tf.layers.flatten(pool)  # 2*2*1024=4096
            print(flatten)
        with tf.variable_scope("fc1") as scope:
            # hidden_Weights1 = tf.Variable(tf.truncated_normal([channel*16, 256], stddev=0.1))  # 45-7040 40-5632
            # hidden_biases1 = tf.Variable(tf.constant(0.1, shape=[256]))
            dense = tf.layers.dense(flatten, units=256, activation=None)
            print(dense)
            # net = tf.add(tf.matmul(flatten, hidden_Weights1), hidden_biases1)
        return dense

    def contrastive_loss(self, model1, model2, y):
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            output_difference = tf.abs(model1 - model2)
            dense = tf.layers.dense(output_difference, units=1, activation=None)
            print(dense)
            y_ = tf.nn.sigmoid(dense, name='distance')
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=dense, labels=y)
            loss = tf.reduce_mean(losses)
        return y_, loss

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
        var_list = tf.global_variables()
        for var in var_list:
            print(var.name)
            if "Siamese/output/dense/kernel:0" == var.name:
                x = var
            if "Siamese/output/dense/bias:0" == var.name:
                b = var
        wx_plus_b = tf.matmul(output_difference, x)+b
        self.test_y_hat = tf.nn.sigmoid(wx_plus_b, name='distance')

if __name__ == '__main__':
    model = Siamese()
    var_list = tf.global_variables()
    for var in var_list:
        print(var)

