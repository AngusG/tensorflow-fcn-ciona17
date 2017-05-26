import tensorflow as tf
import numpy as np
import os
import os.path
from datetime import datetime
import time
import random
import argparse

import matplotlib.pyplot as plt

from models.vgg7_fc6_512_deconv import vgg16

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
        '--savepath', default='/export/mlrg/gallowaa/Documents/conf/tensorflow-fcn-ciona17/')
    parser.add_argument(
        '--plot', help='periodically plot validation progress during training', action="store_true")

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
            vgg = vgg16(args.out, filter_dims_arr[0, :], rgb, keep_prob)
        elif args.model == 's':
            vgg = vgg16(args.out, filter_dims_arr[1, :], rgb, keep_prob)
        else:
            vgg = vgg16(args.out, filter_dims_arr[2, :], rgb, keep_prob)

        logits = vgg.up
        logits = tf.reshape(logits, [-1])

        # binarize the network output

        #prediction = tf.greater_equal(logits, 0.5)
        prediction = tf.cast(tf.greater_equal(
            logits, 0.5, name='thresh'), tf.int32)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())  # v0.12

        init = tf.global_variables_initializer()  # v0.12
        init_locals = tf.local_variables_initializer()  # v0.12

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options={'allow_growth': True}))

        sess.run([init, init_locals])

        if args.restore:
            print('Restoring the network from %s' % os.path.join(args.path, args.restore))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.path, args.restore)))
        else:    
            print('Restoring the network from path %s' % args.path)
            saver.restore(sess, tf.train.latest_checkpoint(args.path))

        print('Running the Network')

        num_batches = 334
        miou_list = np.zeros(int(num_batches / FLAGS.batch_size))

        step = sess.run(global_step)
        print(step)

        image = plt.imread(args.image)

        predimg = sess.run(prediction, feed_dict={keep_prob: 1.0, p_rgb: image})

        if args.plot:

            print(step)

            predimg = predimg.reshape(1, image.shape[0], image.shape[1], 1)
            img2 = np.asarray(predimg)
            img2 = np.squeeze(predimg)
            
            fname =  os.path.join(args.savepath + os.path.basename(args.image)[:-4]) + '_step_' + str(step) + '.png'
            plt.imsave(fname, img2)
