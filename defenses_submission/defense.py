"""The code is to conduct defense against adversarial attacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ConfigParser
import cv2
import numpy as np
import os
import tensorflow as tf
import timeit
import util

import all_models

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory of images",
                    type=str)
parser.add_argument("--output_file", help="output directory of images",
                    type=str)
args = parser.parse_args()

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
# configuration for the defender
try:
    use_augmentation = Config.getboolean('input', 'use_augmentation')
except:
    print('augmentation not found, set to false')
    use_augmentation = False
if use_augmentation:
    crop_ratio = Config.getfloat('input', 'crop_ratio')
model_names = Config.get('model', 'names')
try:
    model_weights = [float(x) for x in Config.get('model', 'weights').split(',')]
    print('using model weights', model_weights)
except:
    model_weights = None
try:
    flip_left_and_right = Config.getboolean('input', 'use_flip')
except:
    flip_left_and_right = False
try:
    use_denoiser = Config.get('defender', 'denoiser')
except:
    use_denoiser = None
if use_denoiser:
    try:
        save_denoised = Config.getboolean('defender', 'save_denoised')
    except:
        save_denoised = False
    if use_denoiser == 'quantization':
        try:
            quantization_bin = Config.getint('defender', 'quantization_bin')
        except:
            quantization_bin = 16
        print('use quantization_bin ', quantization_bin)
else: save_denoised = False

try:
    ensemble_method = Config.get('model', 'ensemble_method')
except:
    ensemble_method = 'prob'
print('using ensemble method ', ensemble_method)

try:
    anti_fgsm = Config.getboolean('model', 'anti_fgsm')
except:
    anti_fgsm = False
print('anti_fgsm setting is ', anti_fgsm)

start_time = timeit.default_timer()

# set up input
batch_shape = [batch_size, image_height, image_width, 3]
if use_augmentation: batch_shape[0] = int(batch_shape[0] / 4)
img = tf.placeholder(tf.float32, shape=batch_shape, name='image')
if use_augmentation:
    if batch_size % 4 != 0:
        raise ValueError('batch_size need to be multiply of 4')
    if crop_ratio >= 1 or crop_ratio <= 0:
        raise ValueError('crop ration need to be btween 0 and 1')
    box_ind = []
    for i in range(batch_size):
        box_ind.append(np.floor(i / 4).astype(np.int))
    boxes = []
    for i in range(int(batch_size / 4)):
        boxes.append([0, 0, crop_ratio, crop_ratio])
        boxes.append([0, 1 - crop_ratio, crop_ratio, 1])
        boxes.append([1 - crop_ratio, 0, 1, crop_ratio])
        boxes.append([1 - crop_ratio, 1 - crop_ratio, 1, 1])
    img_input = tf.image.crop_and_resize(img, np.array(boxes), np.array(box_ind), [image_height, image_width])
else:
    img_input = img

if flip_left_and_right:
    img_input = img_input[:,:,::-1,:]

print('using device ', gpu)
with tf.device(gpu):
    if anti_fgsm:
        _, fgsm_logits, fgsm_vars, fgsm_ckpt = all_models.get_models(img_input, ['inception_v3'])
        fgsm_logits = fgsm_logits[0]
        fgsm_ckpt = fgsm_ckpt[0]
        fgsm_vars = fgsm_vars[0]
        fgsm_loss = tf.reduce_mean(tf.exp(fgsm_logits), axis=1)
        fgsm_grad = tf.gradients(fgsm_loss, img_input)
        img_input = tf.clip_by_value(img_input - 16/256. * tf.sign(fgsm_grad[0]), -1, 1)

    # define model or model ensembles
    all_probs, _, all_model_vars, ckpts = all_models.get_models(img_input, model_names.split('+'))
    if ensemble_method == 'prob':
        probs = util.ensemble(all_probs, weights=model_weights)
        # augmentation
        if use_augmentation:
            probs = tf.reshape(probs, [int(batch_size / 4), 4, -1])
            probs = tf.reduce_mean(probs, axis=1, keep_dims=False)
        # get predicted class label
        predicted_labels = tf.argmax(probs, axis=1)
    elif ensemble_method == 'vote':
        if model_weights is None:
            model_weights = [1 for _ in all_probs]
        all_preds = 0
        for prob, weight in zip(all_probs, model_weights):
            probmax = tf.reduce_max(prob, 1, keep_dims=True)
            y = tf.to_float(tf.equal(prob, probmax))
            preds = y / tf.reduce_sum(y, 1, keep_dims=True)
            all_preds += preds * weight
        if use_augmentation:
            all_preds = tf.reshape(all_preds, [int(batch_size / 4), 4, -1])
            all_preds = tf.reduce_mean(all_preds, axis=1, keep_dims=False)
        predicted_labels = tf.argmax(all_preds, axis=1)
    else:
        raise ValueError('ensemble method %s not found' % (ensemble_method))


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

if anti_fgsm:
    saver = tf.train.Saver(fgsm_vars)
    saver.restore(sess, os.path.join(script_folder, fgsm_ckpt))
    print('weights restored for ' + fgsm_ckpt + '-------------------')

if save_denoised:
    denoised_path = os.path.join(os.path.dirname(args.output_file), 'denoised')
    os.mkdir(denoised_path)
    print('denoised_path: ', denoised_path)

# process
count = 0
every_n = 10
with tf.gfile.Open(args.output_file, 'w') as out_file:
    for filenames, images in util.load_images(args.input_dir, batch_shape, normalize=False, max_number=max_number):
        count += 1
        if count % every_n == 0:
            start_time_loop = timeit.default_timer()
        # perform image denoising
        if not use_denoiser:
            pass
        elif use_denoiser == 'rand_uniform':
            images = images + 16 * np.random.uniform(-1, 1, images.shape)
        elif use_denoiser == 'rand_gaussian':
            images = images + 16 * np.random.normal(0, 0.7, images.shape)
        elif use_denoiser == 'quantization':
            images = np.round(images / quantization_bin) * quantization_bin
        else:
            raise ValueError("unsupported denoising method ", use_denoiser)
        images = util.normalize_image(images)
        if save_denoised:
            util.save_images(images, filenames, denoised_path)
        labels = sess.run(predicted_labels, feed_dict={img: images})
        for filename, label in zip(filenames, labels):
            out_file.write('{0},{1}\n'.format(filename, label))
        if count % every_n == 0:
            elapsed = timeit.default_timer() - start_time_loop
            print('processing speed ' + str(batch_shape[0]/elapsed) + ' samples per second')

elapsed = timeit.default_timer() - start_time
print('total elapsed time ' + str(elapsed) + ' second')
