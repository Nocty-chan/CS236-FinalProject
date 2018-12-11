'''
Trainer for unrestricted generative examples
'''

import tensorflow as tf
import numpy as np

from utils import *
import os
import math

class AttackTrainer(object):
    def __init__(self, sess, params, gan_trainer, classifier_trainer):
        self.params = params
        self.session = sess
        self.gan_trainer = gan_trainer
        self.classifier_trainer = classifier_trainer
        self.gan_trainer.load_checkpoint()
        self.classifier_trainer.load_checkpoint()
        self.build_graph()
        sess.run(tf.variables_initializer(tf.global_variables(scope='unrestricted_adv')))

#         self.load_checkpoint()
        self.images_dir = os.path.join(
            self.params["log_dir"],
            "GAN_adv",
            self.classifier_trainer.params['experiment_name'],
            self.gan_trainer.params['experiment_name'],
            "l1{}_l2{}_lr{}_eps{}_zeps{}".format(
            self.params["lambda1"],
            self.params["lambda2"],
            self.params["lr"],
            self.params["epsilon"],
            self.params["z_eps"]
        ))
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)

    def build_graph(self):
        with tf.variable_scope('unrestricted_adv'):
            self.source_class = tf.placeholder(tf.int32, shape=(None,))

            self.adv_z = tf.get_variable('adv_z',
                                    shape=(self.params["num"], self.gan_trainer.params["z_dim"]),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer)
            self.ref_z = tf.get_variable('ref_z',
                                    shape=(self.params["num"], self.gan_trainer.params["z_dim"]),
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer)
            self._iter = tf.placeholder(tf.float32, shape=(), name="iter")

            # Trainable noise variable
            adv_noise = tf.get_variable('adv_noise', shape=(self.params["num"], 32, 32, 3),
                                        dtype=tf.float32, initializer=tf.zeros_initializer)

            # Generate samples
            self.samples = self.gan_trainer.generator(None, self.source_class, self.gan_trainer.params, noise=self.adv_z)
            _, gan_logits = self.gan_trainer.discriminator(self.samples, self.source_class, self.gan_trainer.params)
            self.samples = tf.transpose(tf.reshape(self.samples, (-1, 3, 32, 32)), (0, 2, 3, 1))
            self.gan_preds = tf.argmax(gan_logits, axis=1) # (num,)
            self.gan_softmax = tf.nn.softmax(gan_logits)

            noise = tf.nn.tanh(adv_noise) * self.params['epsilon']
            self.samples_noise = tf.clip_by_value(self.samples + noise, clip_value_min=-1., clip_value_max=1.0)

            # Classify samples
            logits = self.classifier_trainer.model(0.5 * (self.samples_noise + 1), False, self.classifier_trainer.params)
            self.net_softmax = tf.nn.softmax(logits)
            self.net_preds = tf.argmax(logits, axis=1)

            obj_classes = []
            for i in range(self.params['num_classes']):
                onehot = np.zeros((self.params["num"], self.params['num_classes']), dtype=np.float32)
                onehot[:, i] = 1.0
                obj_classes.append(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot))

            all_cross_entropy = tf.stack(obj_classes, axis=1) # BS, K
            one_hot_source_class = tf.one_hot(self.source_class, self.params['num_classes'])
            all_cross_entropy_modified = (1 - one_hot_source_class) * all_cross_entropy + one_hot_source_class * tf.reduce_max(all_cross_entropy)

            self.min_cross_entropy = tf.reduce_mean(tf.reduce_min(all_cross_entropy_modified, axis=1))
            self.noise_loss = tf.reduce_mean(tf.maximum(tf.square(self.ref_z - self.adv_z) - self.params['z_eps'] ** 2, 0.0))
            self.acgan_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=gan_logits, labels=self.source_class))

            # Loss
            self.loss = self.min_cross_entropy + \
                self.params['lambda1'] * self.noise_loss + \
                self.params['lambda2'] * self.acgan_loss
            with tf.variable_scope("train_ops"):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.params['lr'])
                var = 0.01 / (1. + self._iter) ** 0.55
                grads = optimizer.compute_gradients(self.loss, var_list=[self.adv_z, adv_noise])
                new_grads = []
                for grad, v in grads:
                    if v is not adv_noise:
                        new_grads.append((grad + tf.random_normal(shape=grad.get_shape().as_list(), stddev=tf.sqrt(var)), v))
                    else:
                        new_grads.append((grad / tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3], keepdims=True) + 1e-12), v))
                self.adv_op = optimizer.apply_gradients(new_grads)

            momentum_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_ops'))
            init_op = tf.group(momentum_init, tf.variables_initializer([self.adv_z, adv_noise]))
            with tf.control_dependencies([init_op]):
                self.init_op = tf.group(init_op, tf.assign(self.ref_z, self.adv_z))

            summaries = []
            summaries.append(tf.summary.scalar('Loss', self.loss))
            summaries.append(tf.summary.scalar('Entropy Loss', self.min_cross_entropy))
            # summaries.append(tf.summary.scalar('Target loss', self.min_cross_entropy))
            summaries.append(tf.summary.scalar('Noise Loss', self.noise_loss))
            summaries.append(tf.summary.scalar('Adv Noise', tf.reduce_mean(tf.nn.tanh(adv_noise))))
            summaries.append(tf.summary.scalar('AC_Gan Loss', self.acgan_loss))
            summaries.append(tf.summary.scalar('Gradient/adv_z', tf.reduce_mean(grads[0][0])))
            summaries.append(tf.summary.scalar('Gradient/adv_noise', tf.reduce_mean(grads[1][0])))
            summaries.append(tf.summary.scalar('New_Gradient/adv_z', tf.reduce_mean(new_grads[0][0])))
            summaries.append(tf.summary.scalar('New_Gradient/adv_noise', tf.reduce_mean(new_grads[1][0])))

            self.summary = tf.summary.merge(summaries)

            show_all_variables(scope='unrestricted_adv')

    def generate(self, source_class, target_class, iters):
        '''
        Generates adversarial examples
        source_class: (num,) ND-array of source classes
        target_class: (num,) ND-array of target classes
        iters: int, number of iterations to run SGD on
        '''
        writer = tf.summary.FileWriter(self.images_dir, self.session.graph)
        self.session.run(self.init_op)
        preds_np, probs_np, im_np, original_images, cost_before = \
            self.session.run(
                [self.net_preds, self.net_softmax, self.samples_noise, self.samples, self.loss],
                feed_dict={
                    self.source_class:source_class
                }
            )
        acc = 0.
        adv_acc = 0.
        adv_im_np = []
        original_im_np = []
        print("Initial accuracy on the samples: ", np.sum(preds_np == source_class))
        ###### Using GD for attacking
        # initialize optimizers
        for i in range(iters):
            _, now_cost, pred_np, acgan_pred_np, acgan_probs, images, without_noise, summ = self.session.run(
                [self.adv_op, self.loss, self.net_preds, self.gan_preds, self.gan_softmax, self.samples_noise, self.samples, self.summary],
                feed_dict={
                    self._iter: i,
                    self.source_class: source_class
                })
            ok = np.logical_and(pred_np != source_class, acgan_pred_np == source_class)

            writer.add_summary(summ, i)

            if i % 1 == 0:
                visualize_images(0.5 * (images + 1), path=os.path.join(self.images_dir, "generated_at_iters{}.jpg".format(i)))
                visualize_images(0.5 * (without_noise + 1), path=os.path.join(self.images_dir, "without_noise_images_gan{}.jpg".format(i)))
                print("[*] {}th iter, cost: {}, success: {}/{}".format(i, now_cost, np.sum(ok), self.params["num"]))
                print("[*] Current accuracy on the samples:{}".format(np.mean(pred_np == source_class)))
                print("[*] Current accuracy of AC-GAN: {}".format(np.mean(acgan_pred_np == source_class)))
        adv_preds_np, acgan_preds_np, adv_probs_np, acgan_probs_np, im_np, hidden_z, init_z, cost_after = self.session.run(
            [self.net_preds, self.gan_preds,
                self.net_softmax, self.gan_softmax, self.samples, self.adv_z, self.ref_z, self.loss],
            feed_dict={
                self.source_class: source_class,
            })
        acc += np.sum(preds_np == source_class)
        idx = np.logical_and(adv_preds_np != source_class, acgan_preds_np == source_class)
        adv_acc += np.sum(idx)
        adv_im_np.extend(im_np[idx])
        original_im_np.extend(original_images[idx])
        print("acc: {}, adv_acc: {}, num collected: {}, cost before: {}, cost after: {}".
            format(acc, adv_acc,
                len(adv_im_np), cost_before, cost_after))
        return im_np, original_images

    def get_closest(images):
        train_X = self.classifier_trainer.train_X
        images = np.reshape(-1, 1, 32 * 32 * 3) # N, 1,  -1
        train_X = np.reshape(1, -1, 32 * 32 * 3) # 1, M, -1
        distances = np.sum((images - train_X)**2, axis = -1) # N, M
        indices = np.argmin(distances, axis=1) # N,
        return self.classifier_trainer.train_X[indices]

    def load_checkpoint(self):
        self.gan_trainer.load_checkpoint()
        self.classifier_trainer.load_checkpoint()
