import argparse

import datetime
import sys, random, time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib

FLAGS = None

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num


def mnistModel(X, keep_prob, reuse=False):
    with tf.variable_scope('L1', reuse=reuse):
        L1 = tf.layers.conv2d(X, 64, [3, 3], reuse=reuse)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, keep_prob, True)

    with tf.variable_scope('L2', reuse=reuse):
        L2 = tf.layers.conv2d(L1, 128, [3, 3], reuse=reuse)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, keep_prob, True)

    with tf.variable_scope('L2-1', reuse=reuse):
        L2_1 = tf.layers.conv2d(L2, 128, [3, 3], reuse=reuse)
        L2_1 = tf.layers.max_pooling2d(L2_1, [2, 2], [2, 2])
        L2_1 = tf.layers.dropout(L2_1, keep_prob, True)

    with tf.variable_scope('L3', reuse=reuse):
        L3 = tf.contrib.layers.flatten(L2_1)
        L3 = tf.layers.dense(L3, 1024, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, keep_prob, True)

    with tf.variable_scope('L4', reuse=reuse):
        L4 = tf.layers.dense(L3, 256, activation=tf.nn.relu)

    with tf.variable_scope('LF', reuse=reuse):
        LF = tf.layers.dense(L4, 10, activation=None)

    
    return LF 
 
def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    num_gpus = FLAGS.num_gpus

    print (ps_hosts, worker_hosts, num_gpus)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts })
 
    server = tf.train.Server(cluster, 
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
    


    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % (FLAGS.task_index),
                    cluster=cluster)):

            training_epochs = 10
            batch_size = 1000
            learning_rate = 0.001
            gpu_num = check_available_gpus()
            #gpu_num = 2
            
            mnist = input_data.read_data_sets("./input_data/", one_hot=True)
            # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.float32, [None, 10])
            
            keep_prob = tf.placeholder(tf.float32)

            '''
            losses is the array of each loss from each GPU
            Each GPU learns different input datas each other
            '''
            losses = []
            X_A = tf.split(X, int(gpu_num))
            Y_A = tf.split(Y, int(gpu_num))

            for gpu_id in range(int(gpu_num)):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                        cost = tf.nn.softmax_cross_entropy_with_logits(
                                        logits=mnistModel(X_A[gpu_id], keep_prob, gpu_id > 0),
                                        labels=Y_A[gpu_id])
                        losses.append(cost)
                        
            with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
                loss = tf.reduce_mean(tf.concat(losses, axis=0))
                global_step = tf.train.get_or_create_global_step()
                
                #################################
                # Asynchronous Updating (default)
                train = tf.train.AdamOptimizer(learning_rate).minimize(
                    loss, colocate_gradients_with_ops=True, global_step=global_step)

                #######################
                # Synchronous Updating
                #optimizer = tf.train.SyncReplicasOptimizer(
                #            tf.train.AdamOptimizer(learning_rate), 
                #            replicas_to_aggregate=2,
                #            total_num_replicas=2
                #            )
                #train = optimizer.minimize(loss, global_step=global_step, colocate_gradients_with_ops=True)
                # train = optimizer.minimize(loss, global_step=global_step, colocate_gradients_with_ops=True, aggregation_method=tf.AggregationMethod.ADD_N)
                #sync_replicas_hook = optimizer.make_session_run_hook((FLAGS.task_index == 0))

            # Calculate accuracy
            is_correct = tf.equal(tf.arg_max(mnistModel(X, keep_prob, reuse=True), 1), tf.arg_max(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


        

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.        



        # Supervisor vs MonitoredTrainingSession
        
        # init_op = tf.global_variables_initializer()
        # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
        #                             logdir="/tmp/train_logs",
        #                             init_op=init_op,
        #                             global_step=global_step
        #                             )
        # with sv.managed_session(server.target, 
        #                         config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        #                         ) as sess:


        # MonitoredTrainingSession options        
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # scaffold = tf.train.Scaffold(local_init_op=init_op)
        hooks=[tf.train.StopAtStepHook(last_step=100000)]

        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=(FLAGS.task_index == 0),
                                            checkpoint_dir="tmp/train_logs/distTest",
                                            hooks=hooks,
                                            # hooks=[sync_replicas_hook],
                                            # scaffold=scaffold,
                                            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                                            ) as sess:

            # while not mon_sess.should_stop():
                # mon_sess.run(train)

            # tf.train.start_queue_runners(sess=mon_sess)
            total_batch = int(mnist.train.num_examples / batch_size)
            start_time = datetime.datetime.now()
            tmp_time = start_time
            for epoch in range(training_epochs):
                total_cost = 0

                for i in range(total_batch):
                    # 각 batch size 마다 weight들이 업데이트 됨.
                    # 이 부분에서 서버간 통신을 통해 파라미터 업데이트 해야 함
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    batch_xs = batch_xs.reshape(-1, 28, 28, 1)

                    c, _ = sess.run([loss, train], 
                                            feed_dict={
                                                X: batch_xs,
                                                Y: batch_ys,
                                                keep_prob: 0.7 
                                                }) 
                    total_cost += c
                    # total_cost += c / total_batcha

                elapsed_time = datetime.datetime.now() - tmp_time
                tmp_time = datetime.datetime.now()
                
                print('Epoch: {:2d}'.format(epoch + 1),
                    'cost: {:.9f}'.format(total_cost),
                    'elapsed_time: {}s'.format(elapsed_time))

            print("Learning finished")
            print("--- Training time : {} seconds ---".format(datetime.datetime.now() - start_time))
            # Test the model using test sets
            
            print("Model Accuracy: ", accuracy.eval(session=sess, 
                feed_dict={
                    X: mnist.test.images.reshape(-1, 28, 28, 1) , 
                    Y: mnist.test.labels, 
                    keep_prob:1.0
                    }))
        

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
  # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
