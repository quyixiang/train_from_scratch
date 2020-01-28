#!/usr/bin/env python

import matplotlib

import sys
import tensorflow as tf
import _pickle as pickle
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from VGG_structure.vgg16_build import *

DATASET_NUM = 10000
BATCH = 100
EPOCH = 10


with tf.Session() as sess:
    vgg = VGG16()

    w = tf.Variable(tf.truncated_normal([512,10], 0.0, 1.0) * 0.01, name='w_last')
    b = tf.Variable(tf.truncated_normal([10], 0.0, 1.0) * 0.01, name = 'b_last')

    input = tf.placeholder(tf.float32,[None, 32, 32, 3])

    fmap = vgg.build(input, True)
    predict = tf.nn.softmax(tf.add(tf.matmul(fmap, w), b))
    print(predict.shape)

    #这段没懂
    ans = tf.placeholder(shape=None, dtype=tf.float32)
    ans = tf.squeeze(tf.cast(ans, tf.float32))
    print("----------------")
    tf.nn.con
    print(ans)

    # 交叉熵
    loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(predict), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train_step = optimizer.minimize(loss)

    






    
    







