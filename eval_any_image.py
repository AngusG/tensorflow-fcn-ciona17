import tensorflow as tf
import numpy as np
import os
#import os.path
from datetime import datetime
import time
import random
import argparse

import matplotlib.pyplot as plt

#from skimage import util
from models.vgg7_fc6_512_deconv import vgg
from utils import input_pipeline_xent, input_pipeline_miou, init_3subplot, update_plots

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image', help='path to image to segment')
    parser.add_argument(
        '--ds', help='down sample factor', type=int, default=1)
    parser.add_argument(
        '--path', help='path to checkpoints folder')
    parser.add_argument(
        '--restore', help='specific checkpoint to use (e.g model.ckpt-99000), otherwise use latest')
    parser.add_argument(
        '--gpu', help='the physical ids of GPUs to use')
    parser.add_argument(
        '--out', help='number of semantic classes', type=int, default=1)
    parser.add_argument(
        '--fmt', help='input image format (either rgb or lab)', default='rgb')
    parser.add_argument(
        '--model', help='the model variant (should match checkpoint otherwise will crash)', default='')
    parser.add_argument(
        '--show', help='show the image or just save directly to file', action="store_true")
    parser.add_argument(
        '--allow_growth', help='allow gpu memory growth', action="store_true")
    parser.add_argument(
        '--runs', help='how many forward passes to run for benchmarking', type=int, default=5)

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder(tf.float32)
        mIoU_ph = tf.placeholder(tf.float32)

        p_rgb = tf.placeholder(tf.float32, shape=[None, None, 3], name='p_rgb')
        rgb = tf.expand_dims(p_rgb, 0)
        rgb = rgb * 1.0 / 255.0

        print('rgb shape')
        print(rgb.get_shape())

        filter_dims_arr = np.array([[16, 32, 64, 128, 256, 512],    # vgg#xs
                                    [32, 64, 128, 256, 512, 512],   # vgg#s
                                    [64, 128, 256, 512, 512, 512]]) # vgg#

        if args.model == 'xs':
            vgg = vgg(args.out, filter_dims_arr[0, :], rgb, keep_prob)
        elif args.model == 's':
            vgg = vgg(args.out, filter_dims_arr[1, :], rgb, keep_prob)
        else:
            vgg = vgg(args.out, filter_dims_arr[2, :], rgb, keep_prob)

        logits = vgg.up

        prediction = tf.cast(tf.greater_equal(
            logits, 0.5, name='thresh'), tf.int32)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())  # v0.12

        init = tf.global_variables_initializer()  # v0.12
        init_locals = tf.local_variables_initializer()  # v0.12

        # Start running operations on the Graph.
        # allow_growth - Grow memory usage as needed
        # log_device_placement - To find out which devices your operations and tensors are assigned to
        if args.allow_growth:
            allow_growth = True
        else:
            allow_growth = False

        sess = tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth': allow_growth}))
        #    log_device_placement=FLAGS.log_device_placement, gpu_options={'allow_growth': allow_growth}))

        sess.run([init, init_locals])

        if args.restore:
            print('Restoring the network from %s' % os.path.join(args.path, args.restore))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.path, args.restore)))
        else:    
            print('Restoring the network from path %s' % args.path)
            saver.restore(sess, tf.train.latest_checkpoint(args.path))

        step = sess.run(global_step)
        print('Running network trained to step %d' % step)

        img = plt.imread(args.image)[::args.ds, ::args.ds]
        print(img.shape)
        if args.show:
            plt.imshow(img)
            plt.show()
            plt.figure()

        trials = np.zeros(args.runs)

        for i in range(args.runs):

            m = np.ones((img.shape[0], img.shape[1]))
            r = np.random.randn(img.shape[0], img.shape[1])
            m[r > 3] = 0

            for j in range(img.shape[2]):
                img[:,:,j] = m * img[:,:,j]

            start_time = time.time()
            predimg = sess.run(prediction, feed_dict={keep_prob: 1.0, p_rgb: img})
            trials[i] = time.time() - start_time

            print('Forward pass %d took %f' % (i, trials[i]))

            predimg = predimg.reshape(1, img.shape[0], img.shape[1], args.out)

            if args.show:
                plt.imshow(predimg[0,:,:,1])
                plt.pause(0.1)

        print('Average of %d forward passes took %f, std %f' % (args.runs, np.mean(trials[1:]), np.std(trials[1:])))
