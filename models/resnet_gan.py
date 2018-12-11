import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import functools
import numpy as np
import tensorflow as tf

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs, params, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""

    if ('Discriminator' in name):
        labels = None
    if ('Generator' in name):
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=params['num_classes'])
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, params, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, params, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, params, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs, params):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=params["dim_D"])
    conv_2        = functools.partial(ConvMeanPool, input_dim=params["dim_D"], output_dim=params["dim_D"])
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=params["dim_D"], filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, params, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, params["z_dim"]])
    output = lib.ops.linear.Linear('Generator.Input', params["z_dim"], 4 * 4 * params["dim_G"], noise)
    output = tf.reshape(output, [-1, params["dim_G"], 4, 4])
    output = ResidualBlock('Generator.1', params["dim_G"], params["dim_G"], 3, output, params, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', params["dim_G"], params["dim_G"], 3, output, params, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', params["dim_G"], params["dim_G"], 3, output, params, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output, params)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', params["dim_G"], 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, params["output_dim"]])

def Discriminator(inputs, labels, params, is_training=False):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output, params)
    output = ResidualBlock('Discriminator.2', params["dim_D"], params["dim_D"], 3, output, params, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', params["dim_D"], params["dim_D"], 3, output, params, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', params["dim_D"], params["dim_D"], 3, output, params, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', params["dim_D"], 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if params['dropout'] > 0:
        output = tf.layers.dropout(
            output,
            rate=params["dropout"],
            training=is_training
        )
    output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', params["dim_D"], params["num_classes"], output)
    return output_wgan, output_acgan
