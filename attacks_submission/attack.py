"""The code is to conduct non-targeted adversarial attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ConfigParser
import numpy as np
import os
import timeit
import tensorflow as tf

from itertools import izip
from Queue import Queue
from threading import Thread

import all_models
import util

QUEUE_MAXSIZE = 2000

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory of images",
                    type=str)
parser.add_argument("--output_dir", help="output directory of images",
                    type=str)
parser.add_argument("--max_epsilon", help="maximum perturbation",
                    type=int)
args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

script_folder = os.path.dirname(os.path.abspath(__file__))
Config = ConfigParser.ConfigParser()
Config.read(script_folder + '/config.ini')
try:
    gpu = util.get_device(Config.getint('GPU', 'id'))
except:
    gpu = "/gpu:0"
batch_size = Config.getint('input', 'batch_size')
num_classes = Config.getint('input', 'num_classes')
image_height = Config.getint('input', 'image_height')
image_width = Config.getint('input', 'image_width')
try:
    max_number = Config.getint('input', 'max_number')
except:
    max_number = None
loss_func = Config.get('model', 'loss_func')
sigmoid_perturb = Config.getboolean('model', 'sigmoid_perturb')
if sigmoid_perturb:
    lr = Config.getfloat('model', 'learning_rate')
    itr = Config.getint('model', 'iteration')
    optimizer = Config.get('model', 'optimizer')
    try:
        slope = Config.getfloat('model', 'slope')
    except:
        slope = 1
    if optimizer == 'rmsprop':
        rmsprop_decay = Config.getfloat('model', 'rmsprop_decay')
else:
    try:
        optimizer = Config.get('model', 'optimizer')
    except:
        optimizer = 'sign'
    try:
        itr = Config.getint('model', 'iteration')
    except:
        itr = 1
    try:
        reduce_fold = Config.getfloat('model', 'reduce_fold')
    except:
        reduce_fold = itr
    try:
        rand_start = Config.getboolean('model', 'random_start')
    except:
        rand_start = False
try:
    log_prob = Config.getboolean('logging', 'model_probs')
except:
    log_prob = False

model_names = Config.get('model', 'names')
try:
    model_weights = [float(x) for x in Config.get('model', 'weights').split(',')]
    print('using model weights', model_weights)
except:
    model_weights = None
try:
    ensemble_method = Config.get('model', 'ensemble_method')
except:
    ensemble_method = 'prob'
try:
    emphasize = Config.getint('model', 'emphasize')
except:
    emphasize = None
print('emphasizing ', emphasize)

start_time = timeit.default_timer()
# set up input
eps = 2.0 * args.max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
img = tf.placeholder(tf.float32, shape=batch_shape, name='image')

print('using device ', gpu)
with tf.device(gpu):
    x = tf.Variable(tf.zeros(batch_shape), name='x')
    labels = tf.Variable(tf.zeros([batch_size, num_classes]), name='label')
    x_max = tf.Variable(tf.zeros(batch_shape), name='xmax')
    x_min = tf.Variable(tf.zeros(batch_shape), name='xmin')
    init_ops = [tf.assign(x_max, tf.clip_by_value(img + eps, -1.0, 1.0), name='xmax_init_op'),
                tf.assign(x_min, tf.clip_by_value(img - eps, -1.0, 1.0), name='xmin_init_op')]
    if rand_start:
        img_rand = tf.clip_by_value(img + tf.random_uniform(batch_shape, -eps, eps, seed=128), -1.0, 1.0)
        init_ops.append(tf.assign(x, img_rand, name='x_init_op'))
    else:
        init_ops.append(tf.assign(x, img, name='x_init_op'))
    if sigmoid_perturb:
        diff = tf.Variable(tf.zeros(batch_shape), name='diff')
        img_input = img + eps * (tf.sigmoid(diff * slope) * 2 - 1)
        img_input = tf.clip_by_value(img_input, x_min, x_max)
        init_op = tf.variables_initializer([diff], name='init_op')
    else:
        img_input = x

    # define model or model ensembles
    all_probs, _, all_model_vars, ckpts = all_models.get_models(img_input,
                                                                model_names.split('+'))
    probs = util.ensemble(all_probs, weights=model_weights)

    # get predicted class label
    probmax = tf.reduce_max(probs, 1, keep_dims=True)
    y = tf.to_float(tf.equal(probs, probmax))
    preds = y / tf.reduce_sum(y, 1, keep_dims=True)
    label_op = tf.assign(labels, preds)

    if log_prob:
        maxprob_labels = []
        for prob in all_probs:
            maxprob_labels.append(tf.argmax(prob, 1))

    if ensemble_method == 'prob':
        # define loss function
        if loss_func not in util.__dict__.keys():
            raise ValueError('%s is not a valid loss function.' % loss_func)
        loss = util.__dict__[loss_func](probs, labels)
    elif ensemble_method == 'loss':
        loss = util.ensemble([util.__dict__[loss_func](prob, labels) for prob in all_probs], weights=model_weights)
        if emphasize is not None:
            loss_eph = util.__dict__[loss_func](all_probs[emphasize], labels)
    else:
        raise ValueError(ensemble_method+ ' is not defined')

    # calculate outputs
    if sigmoid_perturb:
        if optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(lr, decay=rmsprop_decay)
        else:
            raise ValueError('optimizer not defined: ', optimizer)
        grad = opt.compute_gradients(loss, var_list=[diff])
        train_op = opt.apply_gradients(grad)
    else:
        grad = tf.gradients(loss, x)
        if optimizer == 'sign':
            new_x = tf.clip_by_value(x - eps / reduce_fold * tf.sign(grad[0]), x_min, x_max)
        elif optimizer == 'norm':
            moments = tf.nn.moments(grad[0], [1, 2, 3], keep_dims=True)
            new_x = tf.clip_by_value(x - eps / reduce_fold * grad[0] / tf.sqrt(moments[1]), x_min, x_max)
        else:
            raise ValueError('optimizer not defined: ', optimizer)
        train_op = tf.assign(x, new_x)
        if emphasize is not None:
            grad = tf.gradients(loss_eph, x)
            if optimizer == 'sign':
                new_x = tf.clip_by_value(x - eps / reduce_fold /1.5 * tf.sign(grad[0]), x_min, x_max)
            elif optimizer == 'norm':
                moments = tf.nn.moments(grad[0], [1, 2, 3], keep_dims=True)
                new_x = tf.clip_by_value(x - eps / reduce_fold * grad[0] / tf.sqrt(moments[1]), x_min, x_max)
            else:
                raise ValueError('optimizer not defined: ', optimizer)
            train_op_eph = tf.assign(x, new_x)
print('computation graph built------------------------------------')

# initialize
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
print('parameters intialized------------------------------------')
for model_vars, ckpt in zip(all_model_vars, ckpts):
    saver = tf.train.Saver(model_vars)
    saver.restore(sess, os.path.join(script_folder, ckpt))
    print('weights restored for ' + ckpt + '-------------------')
elapsed = timeit.default_timer() - start_time
print('loading elapsed time ' + str(elapsed) + ' second')

# process
cache_tbs = Queue(maxsize=QUEUE_MAXSIZE)
# start a thread to handle image saver
t = Thread(target=util.save_images_fast, args=(cache_tbs, args.output_dir,))
t.setDaemon(True)
t.start()
count = 0
every_n = 10
for filenames, images in util.load_images(args.input_dir, batch_shape, max_number=max_number):
    count += 1
    if count % every_n == 0:
        start_time_loop = timeit.default_timer()

    sess.run(init_ops, feed_dict={img: images})
    if itr > 0:
        sess.run([label_op])
    for _ in xrange(itr):
        sess.run([train_op])
        if emphasize is not None:
            sess.run([train_op_eph])
    out_imgs = sess.run(img_input)

    if log_prob:
        maxprob_labels_output = np.vstack(sess.run(maxprob_labels))
        print(maxprob_labels_output)

    for filename, out_img in izip(filenames, out_imgs):
        cache_tbs.put((filename, out_img))
    if count % every_n == 0:
        elapsed = timeit.default_timer() - start_time_loop
        print('processing speed ' + str(batch_shape[0]/elapsed) + ' samples per second')
cache_tbs.put(None)
t.join()

elapsed = timeit.default_timer() - start_time
print('total elapsed time ' + str(elapsed) + ' second')
