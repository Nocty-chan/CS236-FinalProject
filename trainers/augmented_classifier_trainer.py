import tensorflow as tf
from tensorflow import layers
import numpy as np
import os
from trainers.trainer import Trainer
from attacks import *

from utils import visualize_images, show_all_variables, check_folder

class AugmentedTrainer(Trainer):
    '''
    Class to train multiclass classification model using cross entropy loss
    '''
    def __init__(self, sess, model, wgan, params, name="classifier", mode="train"):
        '''
        Init:
        sess: tf.Session()
        model: function that takes in input placeholder, is_training placeholder and params dictionary,
        wgan: trainer for adversarial generator.
        params: dictionary of hyperparameters for the model and training.
        '''
        self.model = model
        self.gan = wgan
        Trainer.__init__(self, sess, params, name, mode)
        self.adversarial_dir = os.path.join(self.checkpoint_dir, "adversarial")
        check_folder(self.adversarial_dir)

    def build_graph(self):
        '''
        Builds computation graph for training and evaluating
        '''
        self.label_clean = tf.placeholder(shape=(None), dtype=tf.int64, name='clean_label')
        self.x_clean = tf.placeholder(
            shape=(None, self.params['img_size'], self.params['img_size'], self.params['img_channel']),
            dtype=tf.float32, name='clean_input'
        )

        self.is_training = tf.placeholder(shape=(), dtype=tf.bool, name='training')
        self.lr = tf.placeholder(shape=(), dtype=tf.float32, name='learning_rate')

        self.logits_clean = self.model(self.x_clean, self.is_training, self.params)
        self.prediction_clean = tf.argmax(self.logits_clean, axis=1)

        self.loss_clean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_clean,
            logits=self.logits_clean,
            name="softmax"))
        self.accuracy_clean = tf.reduce_mean(tf.cast(tf.equal(self.prediction_clean, self.label_clean), tf.float32))

        self.build_graph_adversarial()
        self.num_clean = self.batch_size - self.num_adv
        self.loss = (self.num_clean * self.loss_clean + self.params['lambda'] * self.num_adv * self.loss_adv) \
            / (self.num_clean + self.params['lambda'] * self.num_adv)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="classifier")
        classifier_vars = tf.trainable_variables(scope="classifier")
        summaries = []
        with tf.control_dependencies(update_ops):
            if self.params['optimizer'] == 'Adam':
                print("Using Adam Optimizer")
                self.step = tf.train.AdamOptimizer(
                    learning_rate=self.lr,
                    epsilon=1e-4).\
                    minimize(self.loss, var_list=classifier_vars)

            elif self.params['optimizer'] == 'Momentum':
                print("Using Momentum optimizer with l2 weight decay")
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in classifier_vars])
                self.step = tf.train.MomentumOptimizer(
                    learning_rate=self.lr,
                    momentum=0.9, use_nesterov=True
                ).minimize(
                    self.loss + self.params['weight_decay'] * l2_loss,
                    var_list = classifier_vars
                )
                summaries.append(tf.summary.scalar("L2 Loss", l2_loss))

        summaries.append(tf.summary.scalar("Loss", self.loss))
        summaries.append(tf.summary.scalar("Accuracy", self.accuracy_clean))
        summaries.append(tf.summary.scalar("Adversarial accuracy", self.accuracy_adv))
        summaries.append(tf.summary.scalar("Learning rate", self.lr))
        self.summary = tf.summary.merge(summaries)
        show_all_variables(scope="classifier")
        num_adv = int(self.batch_size * self.params['prop'])
        self.build_unrestricted_generator(num_adv)

        show_all_variables()
        self.init_op = tf.global_variables_initializer()

    def build_graph_adversarial(self):
        self.num_adv = tf.placeholder(shape=(), dtype=tf.float32, name="num_adv")
        self.input_adv = tf.placeholder(
            shape=(None, self.params['img_size'], self.params['img_size'], self.params['img_channel']),
            dtype=tf.float32, name='adv_input'
        )
        self.label_adv = tf.placeholder(shape=(None), dtype=tf.int64, name='adv_label')
        self.logits_adv = self.model(self.input_adv, self.is_training, self.params)

        self.prediction_adv = tf.argmax(self.logits_adv, axis=1)
        self.loss_adv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_adv,
            logits=self.logits_adv,
            name="softmax"))
        self.accuracy_adv = tf.reduce_mean(tf.cast(tf.equal(self.prediction_adv, self.label_adv), tf.float32))

    def train_impl(self, start_epoch, start_batch_id):
        self.sess.run(tf.variables_initializer(tf.global_variables(scope='unrestricted_adv')))
        self.sess.run(self.init_op)
        self.load_checkpoint()
        self.gan.load_checkpoint()
        print("Loaded GAN")
        learning_rate = self.params['lr']
        # DenseNet BC learning schedule
        if self.params['model_name'] in ['densenet']:
            if start_epoch >= (self.params['epochs'] * 0.5):
                learning_rate = learning_rate / 10
            if start_epoch >= (self.params['epochs'] * 0.75):
                learning_rate = learning_rate / 10

        # Sanity check eval Loss ~ 4.6
        self.eval(use_val=self.params['use_val'])
        for epoch in range(start_epoch, self.params['epochs']):
            # Re-permute training data with each epoch
            permutation = np.random.permutation(self.num_data)
            dataX = self.train_X[permutation]
            datay = self.train_y[permutation]

            # Learning rate decay
            if self.params['model_name'] in ['densenet']:
                if epoch == int(self.params['epochs'] * 0.5) or \
                        epoch == int(self.params['epochs'] * 0.75):
                    learning_rate = learning_rate / 10
                    print("New learning rate {}".format(learning_rate))
            for i in range(start_batch_id, self.iters):
                batch_data = dataX[i * self.batch_size:(i + 1) * self.batch_size]
                batch_label = datay[i * self.batch_size:(i + 1) * self.batch_size]
                num_adv = int(self.batch_size * self.params['prop'])
                samples, acc, original_samples, original_acc = self.generate_samples(num_adv, batch_label[:num_adv])
                _, loss_, accuracy_, accuracy_adv_, merge_ = self.sess.run(
                    [self.step, self.loss, self.accuracy_clean, self.accuracy_adv, self.summary],
                    feed_dict={
                        self.x_clean: batch_data[num_adv:],
                        self.label_clean: batch_label[num_adv:],
                        self.input_adv: samples,
                        self.label_adv: batch_label[:num_adv],
                        self.is_training: True,
                        self.lr: learning_rate,
                        self.num_adv: num_adv
                    })
                self.writer.add_summary(merge_, self.counter)
                unrestricted_adv_summ = tf.Summary(value=[
                    tf.Summary.Value(tag="Unrestricted/Original_acc", simple_value=original_acc),
                    tf.Summary.Value(tag='Unrestricted/Acc', simple_value=acc)])
                self.writer.add_summary(unrestricted_adv_summ, self.counter)
                self.writer.flush()
                if i % self.params['log_freq'] == 0:
                    print("Epoch {}, {}/{}".format(epoch, i, self.iters), loss_, accuracy_, accuracy_adv_)
                self.counter += 1

                file_name = str(epoch) + "generated_gan.jpg"
                visualize_images(
                    0.5 * (samples + 1),
                    num_rows=8,
                    path=os.path.join(
                    self.adversarial_dir,
                    file_name
                    )
                )

                file_name = str(epoch) + "original_gan.jpg"
                visualize_images(
                    0.5 * (original_samples + 1),
                    num_rows=8,
                    path=os.path.join(
                    self.adversarial_dir,
                    file_name
                    )
                )

            print("Epoch {}, {}/{}".format(epoch, i, self.iters), loss_, accuracy_, accuracy_adv_)

            self.eval(partial=10, use_val=self.params['use_val'])
            self.save()
            start_batch_id = 0

    def build_unrestricted_generator(self, num_adv):
        with tf.variable_scope('unrestricted_adv'):
            self.source_class = tf.placeholder(tf.int32, shape=(None,))

            self.adv_z = tf.get_variable('adv_z',
                                    shape=(num_adv, self.gan.params["z_dim"]),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer)
            self.ref_z = tf.get_variable('ref_z',
                                    shape=(num_adv, self.gan.params["z_dim"]),
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer)
            self._iter = tf.placeholder(tf.float32, shape=(), name="iter")

            # Trainable noise variable
            adv_noise = tf.get_variable('adv_noise', shape=(num_adv, 32, 32, 3),
                                        dtype=tf.float32, initializer=tf.zeros_initializer)

            # Generate samples
            self.samples = self.gan.generator(None, self.source_class, self.gan.params, noise=self.adv_z)
            _, gan_logits = self.gan.discriminator(self.samples, self.source_class, self.gan.params)
            self.samples = tf.transpose(tf.reshape(self.samples, (-1, 3, 32, 32)), (0, 2, 3, 1))
            self.gan_preds = tf.argmax(gan_logits, axis=1) # (num,)
            self.gan_softmax = tf.nn.softmax(gan_logits)

            noise = tf.nn.tanh(adv_noise) * self.params['epsilon']
            self.samples_noise = tf.clip_by_value(self.samples + noise, clip_value_min=-1., clip_value_max=1.0)

            # Classify samples
            logits = self.model(self.samples_noise, False, self.params)
            self.net_softmax = tf.nn.softmax(logits)
            self.net_preds = tf.argmax(logits, axis=1)

            obj_classes = []
            for i in range(self.params['num_classes']):
                onehot = np.zeros((num_adv, self.params['num_classes']), dtype=np.float32)
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
            self.adv_loss = self.min_cross_entropy + \
                self.params['lambda1'] * self.noise_loss + \
                self.params['lambda2'] * self.acgan_loss
            with tf.variable_scope("gan_train_ops"):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.params['adv_lr'])
                var = 0.01 / (1. + self._iter) ** 0.55
                grads = optimizer.compute_gradients(self.adv_loss, var_list=[self.adv_z, adv_noise])
                new_grads = []
                for grad, v in grads:
                    if v is not adv_noise:
                        new_grads.append((grad + tf.random_normal(shape=grad.get_shape().as_list(), stddev=tf.sqrt(var)), v))
                    else:
                        new_grads.append((grad / tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3], keepdims=True) + 1e-12), v))
                self.adv_op = optimizer.apply_gradients(new_grads)

            momentum_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gan_train_ops'))
            init_op_un = tf.group(momentum_init, tf.variables_initializer([self.adv_z, adv_noise]))
            with tf.control_dependencies([init_op_un]):
                self.gan_init_op = tf.group(init_op_un, tf.assign(self.ref_z, self.adv_z))

            show_all_variables(scope='unrestricted_adv')

    def generate_samples(self, num_adv, source_class):
        '''
        Returns num_adv samples generative adversarially with given source_class
        '''
        original_predictions, original_images = \
            self.sess.run(
                [self.net_preds, self.samples],
                { self.source_class: source_class}
        )
        for i in range(self.params['adv_iters']):
            _, predictions, images, without_noise = self.sess.run(
                [self.adv_op, self.net_preds, self.samples_noise, self.samples],
                feed_dict={
                    self._iter: i,
                    self.source_class: source_class
                })
        images = self.sess.run(self.samples, feed_dict={self.source_class:source_class})
        return images, np.mean(predictions == source_class), original_images, np.mean(original_predictions == source_class)

    def eval(self, partial=None, use_val=True, write=True):
        '''
        Evaluates current model
        partial:(int) if partial is None, evaluate on partial batches
        mode: evaluates on val or test set.
        '''
        testX = self.val_X if use_val else self.test_X
        testy = self.val_y if use_val else self.test_y
        iters = len(testX) // self.batch_size
        if partial is not None:
            # Evaluate on partial batches and randomly permute
            iters = partial
            permutation = np.random.permutation(len(testX))
            testX = testX[permutation]
            testy = testy[permutation]
        total_loss = 0
        total_accuracy = 0

        for i in range(iters):
            batch_data = testX[i * self.batch_size:(i + 1) * self.batch_size]
            batch_label = testy[i * self.batch_size:(i + 1) * self.batch_size]
            loss_, accuracy_ = self.sess.run(
                [self.loss_clean, self.accuracy_clean],
                feed_dict={self.x_clean: batch_data, self.label_clean: batch_label, self.is_training: False})
            total_loss += loss_
            total_accuracy += accuracy_
        total_loss /= iters
        total_accuracy /= iters
        if write:
            test_summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=total_accuracy),
                tf.Summary.Value(tag='Loss', simple_value=total_loss)])
            self.test_writer.add_summary(test_summary, self.counter)
            self.test_writer.flush()
        print("[*] Evaluated model accuracy {}, loss {}".format(total_accuracy, total_loss))
