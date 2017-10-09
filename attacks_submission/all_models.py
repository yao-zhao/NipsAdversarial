import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import inception_v3
from nets import inception_v4
from nets import inception_resnet_v2
from nets import resnet_v2
from nets import densenet


def inference_densenet161(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(densenet.densenet161_arg_scope()):
        logits, _ = densenet.densenet161(x_input,
            dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('densenet161/')]
    return probs, logits, model_vars

def inference_adv_inception_v3(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('adv'):
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('adv/InceptionV3/')]
    return probs, logits, model_vars

def inference_ens3_adv_inception_v3(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('env3_adv'):
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('env3_adv/InceptionV3/')]
    return probs, logits, model_vars

def inference_ens4_adv_inception_v3(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('env4_adv'):
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('env4_adv/InceptionV3/')]
    return probs, logits, model_vars

def inference_ens_adv_inception_resnet_v2(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('adv'):
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('adv/InceptionResnetV2/')]
    return probs, logits, model_vars

def inference_inception_v3(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(x_input,
            dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('InceptionV3/')]
    return probs, logits, model_vars

def inference_inception_v4(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(x_input,
            dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('InceptionV4/')]
    return probs, logits, model_vars

def inference_inception_resnet_v2(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, _ = inception_resnet_v2.inception_resnet_v2(x_input,
            dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('InceptionResnetV2/')]
    return probs, logits, model_vars

def inference_mopi_inception_resnet_v2_1(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('mopi1'):
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('mopi1/InceptionResnetV2/')]
    return probs, logits, model_vars

def inference_mopi_inception_resnet_v2_2(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('mopi2'):
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('mopi2/InceptionResnetV2/')]
    return probs, logits, model_vars

def inference_mopi_inception_resnet_v2_3(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('mopi3'):
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('mopi3/InceptionResnetV2/')]
        return probs, logits, model_vars

def inference_mopi_inception_resnet_v2_4(x_input, dropout_keep_prob=1,  num_classes=1001):
    with tf.variable_scope('mopi4'):
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x_input,
                dropout_keep_prob=dropout_keep_prob, create_aux_logits=False,
                num_classes=num_classes, is_training=False)
            probs = tf.nn.softmax(logits)
            model_vars = [var for var in tf.global_variables() \
                if var.name.startswith('mopi4/InceptionResnetV2/')]
        return probs, logits, model_vars


def inference_resnet_v2_50(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_50(x_input,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('resnet_v2_50/')]
    return probs, logits, model_vars

def inference_resnet_v2_101(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(x_input,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('resnet_v2_101/')]
    return probs, logits, model_vars

def inference_resnet_v2_152(x_input, dropout_keep_prob=1,  num_classes=1001):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_152(x_input,
            num_classes=num_classes, is_training=False)
        probs = tf.nn.softmax(logits)
        model_vars = [var for var in tf.global_variables() \
            if var.name.startswith('resnet_v2_152/')]
    return probs, logits, model_vars

def get_models(x_input, modelnames, num_classes=1001, dropout_keep_prob=1):
    all_models = []
    all_logits = []
    all_model_vars = []
    for name in modelnames:
        probs, logits, model_vars = globals()['inference_'+name](x_input, num_classes=num_classes, dropout_keep_prob=dropout_keep_prob)
        all_models.append(probs)
        all_logits.append(logits)
        all_model_vars.append(model_vars)
    return all_models, all_logits, all_model_vars, ['nets/' + x + '.ckpt' for x in modelnames]
