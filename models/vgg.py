import tensorflow as tf
from tensorflow import layers

def ConvBatch(input, filter_size, is_training, name):
    with tf.variable_scope(name):
        conv2d = layers.conv2d(input, filters=filter_size, kernel_size=3, padding="same", kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm = tf.contrib.layers.batch_norm(
            conv2d,
            activation_fn=tf.nn.relu,
            is_training=is_training)
        return batch_norm

def DenseBatch(input, num_hidden, is_training, name):
    with tf.variable_scope(name):
        dense = layers.dense(input, units=num_hidden, kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm = tf.contrib.layers.batch_norm(
            dense,
            activation_fn=tf.nn.relu,
            is_training=is_training)
        return batch_norm


def VGG_net(x, is_training, params):
    with tf.variable_scope('vgg_16', 'vgg_16', [x]) as sc:
        layer1 = ConvBatch(x, 64, is_training, "layer1")

        layer2 = ConvBatch(layer1, 64, is_training, "layer2")

        maxpool_2 = layers.max_pooling2d(layer2, pool_size=2, strides=2)
        dropout_2 = layers.dropout(maxpool_2, rate=params['dropout'], training=is_training)

        layer3 = ConvBatch(dropout_2, 128, is_training, "layer3")

        layer4 = ConvBatch(layer3, 128, is_training, "layer4")

        maxpool_4 = layers.max_pooling2d(layer4, pool_size=2, strides=2)
        dropout_4 = layers.dropout(maxpool_4, rate=params['dropout'], training=is_training)

        layer5 = ConvBatch(dropout_4, 256, is_training, "layer5")

        layer6 = ConvBatch(layer5, 256, is_training, "layer6")

        layer7 = ConvBatch(layer6, 256, is_training, "layer7")

        maxpool_7 = layers.max_pooling2d(layer7, pool_size=2, strides=2)
        dropout_7 = layers.dropout(maxpool_7, rate=params['dropout'], training=is_training)

        layer8 = ConvBatch(dropout_7, 512, is_training, "layer8")

        layer9 = ConvBatch(layer8, 512, is_training, "layer9")

        layer10 = ConvBatch(layer9, 512, is_training, "layer10")

        maxpool_10 = layers.max_pooling2d(layer10, pool_size=2, strides=2)
        dropout_10 = layers.dropout(maxpool_10, rate=params['dropout'], training=is_training)

        layer11 = ConvBatch(dropout_10, 512, is_training, "layer11")

        layer12 = ConvBatch(layer11, 512, is_training, "layer12")

        layer13 = ConvBatch(layer12, 512, is_training, "layer13")

        maxpool_13 = layers.max_pooling2d(layer13, pool_size=2, strides=2)
        dropout_13 = layers.dropout(maxpool_13, rate=params['dropout'], training=is_training)

        flat_output = layers.flatten(dropout_13)

        dense14 = DenseBatch(flat_output, 4096, is_training, "dense14")
        dropout_14 = layers.dropout(dense14, rate=params['dropout'], training=is_training)

        dense15 = DenseBatch(dropout_14, 4096, is_training, "dense15")

        logits = layers.dense(dense15, 100, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return logits
