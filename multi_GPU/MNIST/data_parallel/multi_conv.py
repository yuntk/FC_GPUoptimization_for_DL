from __future__ import print_function

import input_data
import time
import numpy as np
mnist = input_data.read_data_sets("/home/deepl/FC/multi-GPU/MNIST/data_mnist", one_hot=True)

import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, n):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=n)

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def inference(x, y_, keep_prob):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.variable_scope("conv1") as scope:
    W = _variable_on_cpu(name = 'W', 
                              shape = [5, 5, 1, 32], 
                              initializer= tf.truncated_normal_initializer(stddev=0.1))
    b = _variable_on_cpu(name = 'b', 
                              shape = [32], 
                              initializer= tf.truncated_normal_initializer(stddev=0.1))

    conv1 = tf.nn.relu(conv2d(x_image, W) + b, name=scope.name)
  pool1 = max_pool_2x2(conv1, n = 'pool1')

  with tf.variable_scope("conv2") as scope:
    W = _variable_on_cpu(name = 'W', 
                              shape = [5, 5, 32, 64], 
                              initializer= tf.truncated_normal_initializer(stddev=0.1))
    
    b = _variable_on_cpu(name = 'b', 
                              shape = [64], 
                              initializer= tf.truncated_normal_initializer(stddev=0.1))

    conv2 = tf.nn.relu(conv2d(pool1, W) + b, name = scope.name)

  pool2 = max_pool_2x2(conv2, n = 'pool2')

  # Fully connected layer 1
  pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

  with tf.variable_scope("fc1") as scope:
    W = _variable_on_cpu(name = 'W', 
                              shape = [7 * 7 * 64, 1024], 
                              initializer= tf.truncated_normal_initializer(stddev=0.1))
    b = _variable_on_cpu(name = 'b', 
                              shape = [1024], 
                              initializer= tf.truncated_normal_initializer(stddev=0.1))

    fc1 = tf.nn.relu(tf.matmul(pool2_flat, W) + b, name = scope.name)

  # Dropout
  
  fc1_drop = tf.nn.dropout(fc1, keep_prob)

  with tf.variable_scope("softmax_linear") as scope:
    W = _variable_on_cpu(name = 'W', 
                                shape = [1024,10], 
                                initializer= tf.truncated_normal_initializer(stddev=0.1))
      
    b = _variable_on_cpu(name = 'b', 
                                shape = [10], 
                                initializer= tf.truncated_normal_initializer(stddev=0.1))

    logit = tf.nn.softmax(tf.matmul(fc1_drop, W) + b, name=scope.name)
  return logit


x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')
keep_prob  = tf.placeholder(tf.float32)


opt = tf.train.AdamOptimizer(1e-4)

tower_grads = []
with tf.variable_scope(tf.get_variable_scope()):
  for i in xrange(2):
    with tf.device('/gpu:%d' % i):
      logit = inference(x, y_, keep_prob)

      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(logit), reduction_indices=[1]))

      tf.get_variable_scope().reuse_variables()

      grads = opt.compute_gradients(cross_entropy)

      tower_grads.append(grads)

grad = average_gradients(tower_grads)

train_step = opt.apply_gradients(grad)

correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


# Training steps
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  max_steps = 1000
  latency = []

  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    st = time.time()
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    #print("step = ", step, "\tloss = ", loss)
    latency.append(time.time()-st)
  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

