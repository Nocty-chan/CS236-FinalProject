import tensorflow as tf


def fgsm_tf(x, labels, logits, epsilon=0.2):
    '''
    Returns tensor of adversarial examples constructed with one step of FGSM
    inputs
    tensor x: (BS, H, W, C) input image
    tensor labels: (BS,) ground-truth labels
    tensor logits: (BS, K) output logits from the model
    tensor epsilon: () attack perturbation
    '''
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name="adversarial_loss")
    dy_dx = tf.gradients(losses, x)[0] # (BS, H, W, C)
    adv = tf.clip_by_value(x + epsilon * tf.sign(dy_dx), 0, 1)
    return tf.stop_gradient(adv)

def fgsm_tf_11(x, logits, epsilon=0.2):
    '''
    Returns tensor of adversarial examples constructed with one step of FGSM
    towards the less likely label.
    inputs
    tensor x: (BS, H, W, C) input image
    tensor logits: (BS, K) output logits from the model
    tensor epsilon: () attack perturbation
    '''
    target = tf.cast(tf.argmin(logits, axis=-1), tf.int64)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target,
        logits=logits,
        name="adversarial_loss")
    dy_dx = tf.gradients(losses, x)[0] # (BS, H, W, C)
    adv = tf.clip_by_value(x - epsilon * tf.sign(dy_dx), 0, 1)
    return tf.stop_gradient(adv)
