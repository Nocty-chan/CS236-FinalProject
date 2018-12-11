import numpy as np
import tensorflow as tf


def BottleneckLayer(x, is_training, params, name):
    with tf.variable_scope(name):
        # print("Bottleneck layer input ", x.get_shape())
        batchnorm = tf.layers.batch_normalization(
            inputs=x,
            scale=False,
            training=is_training
        )
        nonlin = tf.nn.relu(batchnorm)
        conv2d = tf.layers.conv2d(
            inputs=nonlin,
            filters=4 * params['growth_rate'],
            kernel_size=1,
            padding='same',
            use_bias=False
        )
        dropout_ = tf.layers.dropout(
            inputs=conv2d,
            rate=params['dropout'],
            training=is_training
        )
        return dropout_


def TransitionLayer(x, num_filters, is_training, params, name):
    with tf.variable_scope(name):
        # print("Transition Layer input", x.get_shape())
        batchnorm = tf.layers.batch_normalization(
            inputs=x,
            scale=False,
            training=is_training
        )
        nonlin = tf.nn.relu(batchnorm)
        conv2d = tf.layers.conv2d(
            inputs=nonlin,
            filters=int(params['theta'] * num_filters),
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=tf.nn.relu
        )
        dropout_ = tf.layers.dropout(
            inputs=conv2d,
            rate=params['dropout'],
            training=is_training
        )
        result = tf.layers.average_pooling2d(dropout_, pool_size=2, strides=2)
        # print("Transition Layer output", result.get_shape())
        return result


def ConvLayer(x, is_training, params, name):
    with tf.variable_scope(name):
        # print("Conv layer input", name, x.get_shape())
        bottled = BottleneckLayer(x, is_training, params, name + "_bottleneck")
        batchnorm = tf.layers.batch_normalization(
            inputs=bottled,
            scale=False,
            training=is_training
        )
        nonlin = tf.nn.relu(batchnorm)
        conv2d = tf.layers.conv2d(
            inputs=nonlin,
            filters=params['growth_rate'],
            kernel_size=3,
            padding='same',
            use_bias=False
        )
        result = tf.layers.dropout(
            inputs=conv2d,
            rate=params['dropout'],
            training=is_training
        )
        return result


def BlockLayer(x, num_filters, is_training, params, name):
    N = int((params['depth'] - 4) / 3) // 2
    # print("Layer depth: ", N)
    total_filters = num_filters
    with tf.variable_scope(name):
        inputs = [x]
        for i in range(N):
            result = ConvLayer(tf.concat(inputs, axis=3, name=name + str(i)), is_training, params, "convd_" + str(i))
            inputs.append(result)
            total_filters += params['growth_rate']
        return tf.concat(inputs, axis=3, name=name + str(N)), total_filters


def dense_net(x, is_training, params):
    '''
    x: Tensor (None, H, C, W)
    is_training: Bool tensor ()
    params: dictionary of parameters
    Returns Tensor (None, params['num_classes'])
    '''
    with tf.variable_scope('densenet'):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=2 * params['growth_rate'],
            kernel_size=3,
            padding='same',
            use_bias=False
        )
        block1, num_filters = BlockLayer(conv1, 2 * params['growth_rate'], is_training, params, 'block1')
        print("Block1 output", block1.get_shape(), num_filters)
        trans1 = TransitionLayer(block1, num_filters, is_training, params, 'transition1')
        print("transition1 output", trans1.get_shape())

        block2, num_filters = BlockLayer(trans1, int(params['theta'] * num_filters), is_training, params, 'block2')
        print("Block2 output", block2.get_shape(), num_filters)
        trans2 = TransitionLayer(block2, num_filters, is_training, params, 'transition2')
        print("transition2 output", trans2.get_shape())

        block3, num_filters = BlockLayer(trans2, int(params['theta'] * num_filters), is_training, params, 'block3')
        print("Block3 output", block3.get_shape(), num_filters)

        lastBatch = tf.layers.batch_normalization(
            inputs=block3,
            scale=False,
            training=is_training
        )
        nonlin = tf.nn.relu(lastBatch)
        global_pool = tf.layers.average_pooling2d(nonlin, pool_size=8, strides=8)
        logits = tf.layers.dense(
            inputs=tf.layers.flatten(global_pool),
            units=params['num_classes'],
            activation=None
        )
    return logits
