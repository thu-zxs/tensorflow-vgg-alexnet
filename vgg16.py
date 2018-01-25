from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import functools
import random
import os, sys, cv2
import re
import pickle

def doublewrap(function):
    """ A decorator decorator,
        allowing to use the decorator to be used without parentheses 
        if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """ A decorator for functions that define Tensorflow operations. 
        The wrapped function will only be executed once. Subsequent calls 
        to it will directly return the result so that operations are added 
        to the graph only once.
        The operations added by the function live within a tf.variable_scope().
        If this decorator is used with arguments, they will be forwarded to the 
        variable scope. The scope name defaults to the name of the wrapped function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def lazy_property(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class VGG16:

    def __init__(self, input_image, target, init_lr, keep_prob, rgb_mean=np.array([116.779, 123.68, 103.939],dtype=np.float32), wd=0.001):
        self.input_image = input_image
        self.target = target
        self.rgb_mean = rgb_mean
        self.keep_prob = keep_prob
        self.init_lr = init_lr
        self.global_step = tf.Variable(0, trainable=False)
        self.weight_decay_factor = wd
        self.predict
        self.weight_decay_loss
        self.cross_entropy_loss
        self.loss
        self.lr
        self.optimize
        self.corrects
        self.accuracy
        self.summary

    def conv(self, input_tensor, name, kh, kw, channel_out, dh=1, dw=1, activation_fn=tf.nn.relu):
        channel_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('W',
                [kh, kw, channel_in, channel_out], 
                tf.float32, 
                initializer=tf.contrib.layers.xavier_initializer()
                #initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            tf.add_to_collection('weights', weights)
            biases = tf.get_variable("b",
                [channel_out],
                tf.float32,
                tf.constant_initializer(0.001)
            )
            conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
            activation = activation_fn(tf.nn.bias_add(conv, biases))
        return activation

    def fully_connected(self, input_tensor, name, channel_out):
        channel_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('W',
                [channel_in, channel_out],
                tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
                #initializer=tf.truncated_normal_initializer()
            )
            tf.add_to_collection('weights', weights)
            biases = tf.get_variable("b",
                [channel_out],
                tf.float32,
                tf.constant_initializer(0.0001)
            )
            logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return logits

    def pool(self, input_tensor, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name
        )

    @define_scope
    def predict(self):
        # assuming 224x224x3 input_tensor
        n_classes = self.target.get_shape()[-1]

        # define image mean
        mu = tf.constant(self.rgb_mean, name="rgb_mean")

        # subtract image mean
        #net = 1.0/255*tf.sub(self.input_image, mu, name="input_mean_centered")
        #net = tf.sub(self.input_image, mu, name="input_mean_centered")
        net = self.input_image

        # block 1 -- outputs 112x112x64
        with tf.variable_scope('block1-o-112x112x64'):
            net = self.conv(net, name="conv1_1", kh=3, kw=3, channel_out=64)
            net = self.conv(net, name="conv1_2", kh=3, kw=3, channel_out=64)
            net = self.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

        # block 2 -- outputs 56x56x128
        with tf.variable_scope('block2-o-56x56x128'):
            net = self.conv(net, name="conv2_1", kh=3, kw=3, channel_out=128)
            net = self.conv(net, name="conv2_2", kh=3, kw=3, channel_out=128)
            net = self.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

        # # block 3 -- outputs 28x28x256
        with tf.variable_scope('block3-o-28x28x256'):
            net = self.conv(net, name="conv3_1", kh=3, kw=3, channel_out=256)
            net = self.conv(net, name="conv3_2", kh=3, kw=3, channel_out=256)
            net = self.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 14x14x512
        with tf.variable_scope('block4-o-14x14x512'):
            net = self.conv(net, name="conv4_1", kh=3, kw=3, channel_out=512)
            net = self.conv(net, name="conv4_2", kh=3, kw=3, channel_out=512)
            net = self.conv(net, name="conv4_3", kh=3, kw=3, channel_out=512)
            net = self.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

        # block 5 -- outputs 7x7x512
        with tf.variable_scope('block5-o-7x7x512'):
            net = self.conv(net, name="conv5_1", kh=3, kw=3, channel_out=512)
            net = self.conv(net, name="conv5_2", kh=3, kw=3, channel_out=512)
            net = self.conv(net, name="conv5_3", kh=3, kw=3, channel_out=512)
            net = self.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

        # flatten
        #flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, 7*7*512], name="flatten")

        # fully connected
        net = self.fully_connected(net, name="fc6", channel_out=4096)
        net = tf.nn.dropout(tf.nn.relu(net), self.keep_prob)
        self.embedding = self.fully_connected(net, name="fc7", channel_out=4096)
        net = tf.nn.dropout(tf.nn.relu(self.embedding), self.keep_prob)
        net = self.fully_connected(net, name="fc8", channel_out=n_classes)
        return tf.nn.softmax(net)

    @define_scope
    def weight_decay_loss(self):
        weights_norm = tf.reduce_sum(
            input_tensor=self.weight_decay_factor*tf.stack(
                [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
            ),
            name='weights_norm'
        )
        tf.add_to_collection('losses', weights_norm)
        return weights_norm

    @define_scope
    def cross_entropy_loss(self):
        #epsilon = tf.constant(0, dtype=tf.float32)
        cross_entropy = -tf.reduce_sum(self.target*tf.log(tf.clip_by_value(self.predict, 1e-10, 1.0)))
        tf.add_to_collection('losses', cross_entropy)
        return cross_entropy

    @define_scope
    def loss(self):
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    @define_scope
    def lr(self):
        boundaries = [4000, 6000, 7000]
        values = [self.init_lr, self.init_lr/10, self.init_lr/50, self.init_lr/100]
        lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
        return lr

    @define_scope
    def optimize(self):
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        #optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        return optimizer.minimize(self.loss)
        #tvars = tf.trainable_variables()
        #grads,_ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy_loss, tvars), 1)
        #return optimizer.apply_gradients(
        #    zip(grads, tvars),
        #    global_step = self.global_step
        #)

    @define_scope
    def corrects(self):
        corrects = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.predict, 1))
        return corrects

    @define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.corrects, tf.float32))
        
    @define_scope
    def summary(self):
        loss_summary = tf.summary.scalar('loss', self.cross_entropy_loss)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        summary = tf.summary.merge([loss_summary, accuracy_summary])#, rnn_summary])
        return summary



