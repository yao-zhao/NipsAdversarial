"""Utility functions used in the project.

This code includes data io, nn loss functions, and ensemble method.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def get_device(gpu_id):
    device = '/gpu:' + str(gpu_id)
    return device
    # local_device_protos = device_lib.list_local_devices()
    # devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    # if device in devices:
    #     return device
    # else: return devices[0]


# data io ----------------------------------------------------------------------

def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


def load_images(input_dir, batch_shape, normalize=True, max_number=None):
    if normalize:
        images_template = np.zeros(batch_shape)
    else:
        images_template = np.zeros(batch_shape, dtype=np.uint8)
    images = images_template
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    image_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
    if max_number:
        image_files = image_files[0:max_number]
    for filepath in image_files:
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB'))
        if normalize:
            images[idx, :, :, :] = normalize_image(image)
        else:
            images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = images_template
            idx = 0
    if idx > 0:
        yield filenames, images


def normalize_image(image):
    return image.astype(np.float32) / 255.0 * 2.0 - 1.0


# This function is to be deprecated. Use multithread image saver instead.
def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def save_images_fast(cache, output_dir):
    while True:
        item = cache.get()
        if item is None:
            return
        filename, image = item
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((image + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


# loss function  ----------------------------------------------------------------------
def nontarget_log_loss(probs, labels):
    with tf.name_scope('nontarget_log_loss'):
        return tf.reduce_sum(tf.multiply(tf.log(probs), labels))

def target_log_loss(probs, labels):
    with tf.name_scope('target_log_loss'):
        return -tf.reduce_sum(tf.multiply(tf.log(probs), labels))

def target_log_loss_soft(probs, labels, soft_label=0.1, num_classes=1001):
    with tf.name_scope('target_log_loss_soft'):
        return -tf.reduce_sum(tf.multiply(tf.log(probs),
            labels*(1-soft_label)+soft_label/num_classes))

def nontarget_log_loss_second_largest_as_target(probs, labels):
    with tf.name_scope('nontarget_log_loss_second_largest_as_target'):
        vals, _ = tf.nn.top_k(probs, k=2, sorted=True)
        probmax2 = tf.slice(vals, [0, 1], [-1, 1])
        y = tf.to_float(tf.equal(probs, probmax2))
        preds = y / tf.reduce_sum(y, 1, keep_dims=True)
        return target_log_loss(probs, preds)


def nontarget_log_loss_smallest_as_target(probs, labels):
    with tf.name_scope('nontarget_log_loss_smallest_as_target'):
        probmin = tf.reduce_min(probs, 1, keep_dims=True)
        y = tf.to_float(tf.equal(probs, probmin))
        preds = y / tf.reduce_sum(y, 1, keep_dims=True)
        return target_log_loss(probs, preds)


def nontarget_log_loss_increase_resprobs_l2(probs, labels):
    with tf.name_scope('nontarget_log_loss_increase_resprobs_l2'):
        l2_norm_sq = tf.reduce_sum(tf.multiply(probs ** 2, 1 - labels), axis=1)
        l2_norm_log = tf.log(l2_norm_sq)
        return -tf.reduce_sum(l2_norm_log)


# model ensemble ----------------------------------------------------------------------

def ensemble(models, weights=None):
    if not weights:
        weights = [1 for model in models]
    with tf.name_scope('ensemble'):
        ensemble_probs = 0
        for model, weight in zip(models, weights):
            ensemble_probs += model * weight
    return ensemble_probs
