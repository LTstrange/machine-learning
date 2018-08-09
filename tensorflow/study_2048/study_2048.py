#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LTstrange'

import tensorflow as tf
import numpy as np
from _2048_ import *

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

fs = tf.placeholder(tf.float32,[None,16])
loss = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(fs,16,8,tf.nn.relu)
l2 = add_layer(l1,8,8,tf.nn.relu)
prediction = add_layer(l2,8,4,tf.nn.softmax)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

field,score = init()
inputs = np.array([])




