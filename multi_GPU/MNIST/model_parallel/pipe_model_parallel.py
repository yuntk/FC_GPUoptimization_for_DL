#!/usr/bin/env python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import time
import threading
import pickle
from Queue import Queue

max_step = 500
b_size = 50
q = Queue(1000)
mnist = input_data.read_data_sets("/home/deepl/Desktop/mnist/data/", one_hot=True)
du = []
ac_t = []
st = time.time()

#x = tf.placeholder("float", [None, 784])
#y_ = tf.placeholder("float", [None,10])

with tf.Graph().as_default():
 class conv_layer(threading.Thread):
  def run(self):
   with tf.device('/gpu:0'):
	x = tf.placeholder("float", [None, 784])
	y_ = tf.placeholder("float", [None,10])
	x_image = tf.reshape(x,[-1,28,28,1])

	c_W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = 0.1))
	c_b1 = tf.Variable(tf.constant(0.1, shape = [32]))

	conv1 = tf.nn.relu(tf.nn.conv2d(x_image, c_W1, strides = [1,1,1,1], padding = 'SAME') + c_b1)
	pool1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	
	c_W2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1))
	c_b2 = tf.Variable(tf.constant(0.1, shape = [64]))
	
	conv2 = tf.nn.relu(tf.nn.conv2d(pool1,c_W2, strides = [1,1,1,1], padding = 'SAME') + c_b2)
	pool2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	
	for i in range(max_step):
		st_st = time.time()
		batch_xs, batch_ys = mnist.train.next_batch(b_size)
		conv_res= sess.run(flatten, feed_dict = {x: batch_xs, y_: batch_ys})
		q.put([batch_ys, conv_res])
		duration = time.time() - st_st
		#print(sess.run(t_vars))

 class local_layer(threading.Thread):
  def run(self):
     with tf.device('/gpu:1'):
	#x = tf.placeholder("float", [None, 784])
	y_ = tf.placeholder("float", [None,10])
	fl = tf.placeholder("float", [None, 3136])
	
	W1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev = 0.1))
	b1 = tf.Variable(tf.constant(0.1, shape = [1024]))
	
	local1 = tf.nn.relu(tf.matmul(fl, W1) + b1)
	
	W2 = tf.Variable(tf.truncated_normal([1024, 10], stddev = 0.1))
	b2 = tf.Variable(tf.constant(0.1, shape = [10]))
	
	y = tf.nn.softmax(tf.matmul(local1,W2) + b2)
	
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	opt = tf.train.AdamOptimizer(1e-4)

	grad = opt.compute_gradients(cross_entropy)
	
	train_step = opt.apply_gradients(grad)
	
	init = tf.initialize_all_variables()

	sess1 = tf.Session()
	sess1.run(init)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	for i in range(max_step):
	  st_st = time.time()
 	  batch_ys, conv_res = q.get()
	  sess1.run(train_step, feed_dict={y_: batch_ys, fl: conv_res})
	  duration = time.time() - st_st	
	  if i % 10 == 0:
		acc = sess1.run(accuracy, feed_dict={y_: batch_ys, fl: conv_res})
		du.append(time.time() - st)
		ac_t.append(acc)
		#print(duration)		
	        print('Step : %d, Accuracy: %f' % (i, acc))
		#print(acc)

 co = conv_layer()
 lo = local_layer()

 co.start()
 lo.start()

