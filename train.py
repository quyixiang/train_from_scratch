#!/usr/bin/env python

import numpy as np
import datetime
import math
import time
import matplotlib.pyplot as plt

import element as ele
from VGG_structure.vgg16_build import *

DATASET_NUM = 10000
BATCH = 100
EPOCH = 10

with tf.Session() as sess:
    vgg = VGG16()

    w = tf.Variable(tf.truncated_normal([514, 2], 0.0, 1.0) * 0.01, name='w_last')
    b = tf.Variable(tf.truncated_normal([2], 0.0, 1.0) * 0.01, name='b_last')

    aug = tf.placeholder(tf.float32, [None, 2])
    input = tf.placeholder(tf.float32, [None, 32, 32, 3])
    fmap = vgg.build(input, True)

    # 在famp这里加参数 famp=(?, 512) + aug=(?,2)
    fmap_new = tf.concat([fmap, aug], axis=1)
    print(fmap_new)
    predict = tf.nn.softmax(tf.add(tf.matmul(fmap_new, w), b))
    # predict = tf.nn.softmax(tf.stack([tf.add(tf.matmul(fmap, w), b), aug]))

    ans = tf.placeholder(shape=None, dtype=tf.float32)
    ans = tf.squeeze(tf.cast(ans, tf.float32))

    # 交叉熵
    loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(predict), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train_step = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    train_images, train_labels, train_augs, test_images, test_labels, test_augs = ele.load_data((-2.0, -1.6), 1., 1000,
                                                                                                1000)

    print('\nSTART LEARNING')
    print('==================== ' + str(datetime.datetime.now()) + ' ====================')

    lossbox = []
    for e in range(EPOCH):
        for b in range(math.ceil(DATASET_NUM / BATCH)):
            batch, labels, augs = ele.get_next_batch(len(train_labels), BATCH, train_images, train_labels, train_augs, test_images,
                                               test_labels, test_augs)
            sess.run(train_step, feed_dict={input: batch, ans: labels, aug: augs})

            print('Batch: %3d' % int(b + 1) + ', \tLoss: ' + str(
                sess.run(loss, feed_dict={input: batch, ans: labels, aug: augs})))

            if (b + 1) % 100 == 0:
                print('============================================')
                print('START TEST')

                images, labels, augs = ele.get_next_batch(len(train_labels), BATCH, train_images, train_labels, train_augs, test_images,
                                                    test_labels, test_augs, is_training=False)
                result = sess.run(predict, feed_dict={input: images ,aug: augs})

                correct = 0
                total = 100

                for i in range(len(labels)):
                    pred_max = result[i].argmax()
                    ans_ = labels[i].argmax()

                    if ans_ == pred_max:
                        correct += 1
                print('Accuracy: ' + str(correct) + ' / ' + str(total) + ' = ' + str(correct / total))

                print('END TEST')
                print('============================================')

            time.sleep(0.01)

        lossbox.append(sess.run(loss, feed_dict={input: batch, ans: labels, aug: augs}))
        print('========== Epoch: ' + str(e + 1) + ' END ==========')

    print('==================== ' + str(datetime.datetime.now()) + ' ====================')
    print('\nEND LEARNING')

    # parameter saver
    saver = tf.train.Saver()
    saver.save(sess, './params.ckpt')

    # plot
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.array(range(EPOCH)), lossbox)
    plt.show()
    plt.savefig("out.png")

