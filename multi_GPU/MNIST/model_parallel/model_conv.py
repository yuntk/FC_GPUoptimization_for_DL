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
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def inference(x, y_, keep_prob):
 with tf.device('/gpu:0'):
  # Input layer
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # Convolutional layer 1
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
 with tf.device('/gpu:1'):
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

#x_d = [x[:50], x[50:]]
#y_d = [y[:50], y[50:]]

opt = tf.train.AdamOptimizer(1e-4)

with tf.variable_scope(tf.get_variable_scope()):
  logit = inference(x, y_, keep_prob)

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(logit), reduction_indices=[1]))

  tf.get_variable_scope().reuse_variables()

  grads = opt.compute_gradients(cross_entropy)


train_step = opt.apply_gradients(grads)


#tf.add_to_collection('losses', cross_entropy)

correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  max_steps = 1000
  latency = []

  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    st = time.time()
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    #print("step = ", step, "\tloss = ", loss, "\tstep/sec = ", time.time()-st)
    latency.append(time.time()-st)
    if (step % 10) == 0:
      print("step = ",step, "\tloss = ",loss , "\t acc = ",  sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}), "\tstep/sec = ", np.average(latency))
      latency = []
  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

