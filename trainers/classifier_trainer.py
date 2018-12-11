import tensorflow as tf
from tensorflow import layers
import numpy as np
import os
from trainers.trainer import Trainer
from attacks import *

from utils import visualize_images, show_all_variables, check_folder

class SGDTrainer(Trainer):
    '''
    Class to train multiclass classification model using cross entropy loss
    '''
    def __init__(self, sess, model, generator, params, name="classifier", mode="train"):
        '''
        Init:
        sess: tf.Session()
        model: function that takes in input placeholder, is_training placeholder and params dictionary,
        adv_trainer: trainer for adversarial generator, must be not None is adversarial_mode == "gan"
        returns a tensor containing the logits.
        params: dictionary of hyperparameters for the model and training.
        '''
        self.model = model
        self.generator = generator
        Trainer.__init__(self, sess, params, name, mode)
        self.adversarial_dir = os.path.join(self.checkpoint_dir, "adversarial")
        check_folder(self.adversarial_dir)

    def build_graph_adversarial(self):
        '''
        Build graph to compute adversarial examples of the model
        '''
        self.input_adv = tf.placeholder(
            shape=(None, self.params['img_size'], self.params['img_size'], self.params['img_channel']),
            dtype=tf.float32, name='adv_input'
        )
        if self.params["adversarial_mode"] == "fgsm":
            self.epsilon_attack = tf.placeholder(shape=(), dtype=tf.float32, name='espilon_attack')
            self.original_logits = self.model(self.input_adv, False, self.params)

            self.x_adv = fgsm_tf_11(self.input_adv, self.original_logits, self.epsilon_attack)
        elif self.params["adversarial_mode"]=="gan":
            self.x_adv = self.input_adv

        self.label_adv = tf.placeholder(shape=(None), dtype=tf.int64, name='adv_label')
        self.logits_adv = self.model(self.x_adv, self.is_training, self.params)

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

        if self.params['adversarial_mode'] is not None:
            self.build_graph_adversarial()
            self.prediction_adv = tf.argmax(self.logits_adv, axis=1)
            self.loss_adv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_adv,
                logits=self.logits_adv,
                name="softmax"))
            self.accuracy_adv = tf.reduce_mean(tf.cast(tf.equal(self.prediction_adv, self.label_adv), tf.float32))
            tf.summary.scalar("Adversarial Accuracy", self.accuracy_adv)
            num_adv = int(self.batch_size * self.params['prop'])
            num_clean = self.batch_size - num_adv
            self.loss = (num_clean * self.loss_clean + self.params['lambda'] * num_adv * self.loss_adv) \
                / (num_clean + self.params['lambda'] * num_adv)
        else:
            self.loss = self.loss_clean

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier_vars = tf.trainable_variables(scope="classifier")
        with tf.control_dependencies(update_ops):
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in classifier_vars])
            self.step = tf.train.MomentumOptimizer(
                learning_rate=self.lr,
                momentum=0.9, use_nesterov=True
            ).minimize(
                self.loss + self.params['weight_decay'] * l2_loss,
                var_list = classifier_vars
            )
            tf.summary.scalar("L2 Loss", l2_loss)

        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Accuracy", self.accuracy_clean)
        tf.summary.scalar("Learning rate", self.lr)
        self.summary = tf.summary.merge_all()
        show_all_variables(scope="classifier")
        self.init_op = tf.global_variables_initializer()

    def train_impl(self, start_epoch, start_batch_id):
        if self.generator is not None:
            self.generator.load_checkpoint()

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

            for i in range(start_batch_id, self.iters + 1):
                if i == self.iters and i * self.batch_size < len(dataX):
                    batch_data = dataX[i * self.batch_size:]
                    batch_label = datay[i * self.batch_size:]
                elif i < self.iters:
                    batch_data = dataX[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_label = datay[i * self.batch_size:(i + 1) * self.batch_size]

                num_adv = int(len(batch_data) * self.params['prop'])
                # Replace first half by Generated Adversarial examples
                if self.params['adversarial_mode'] == "fgsm":
                    epsilon_attack = np.random.uniform(low=1e-4, high=self.params['epsilon'])
                    _, loss_, accuracy_, accuracy_adv_, merge_, generated_adv = self.sess.run(
                        [self.step, self.loss, self.accuracy_clean, self.accuracy_adv, self.summary, self.x_adv],
                        feed_dict={
                            self.x_clean: batch_data[num_adv:],
                            self.label_clean: batch_label[num_adv:],
                            self.input_adv: batch_data[:num_adv],
                            self.label_adv: batch_label[:num_adv],
                            self.is_training: True,
                            self.lr: learning_rate,
                            self.epsilon_attack: epsilon_attack
                        })
                    epsilon_summary = tf.Summary(value=[
                        tf.Summary.Value(tag='Epsilon attack', simple_value=epsilon_attack)])
                    self.writer.add_summary(epsilon_summary, self.counter)
                    if i % self.params['log_freq'] == 0:
                        print("Epoch {}, {}/{}".format(epoch, i, self.iters), epsilon_attack)

                elif self.params['adversarial_mode'] in ["gan", "pixel"]:
                    samples = self.generator.generate(batch_label[:num_adv], num_adv)
                    _, loss_, accuracy_, accuracy_adv_, merge_ = self.sess.run(
                        [self.step, self.loss, self.accuracy_clean, self.accuracy_adv, self.summary],
                        feed_dict={
                            self.x_clean: batch_data[num_adv:],
                            self.label_clean: batch_label[num_adv:],
                            self.input_adv: samples,
                            self.label_adv: batch_label[:num_adv],
                            self.is_training: True,
                            self.lr: learning_rate
                        })
                    generated_adv =  0.5 * (samples + 1)
                else:
                    _, loss_, accuracy_, merge_ = self.sess.run(
                        [self.step, self.loss, self.accuracy_clean, self.summary],
                        feed_dict={
                            self.x_clean: batch_data,
                            self.label_clean: batch_label,
                            self.is_training: True,
                            self.lr: learning_rate
                        })
                    accuracy_adv_ = None
                self.writer.add_summary(merge_, self.counter)
                self.writer.flush()
                if i % self.params['log_freq'] == 0:
                    print("Epoch {}, {}/{}".format(epoch, i, self.iters), loss_, accuracy_, accuracy_adv_)
                self.counter += 1

            print("Epoch {}, {}/{}".format(epoch, i, self.iters), loss_, accuracy_, accuracy_adv_)

            if self.params['adversarial_mode'] is not None:
                if self.params['adversarial_mode'] == "fgsm":
                    file_name = str(epoch) + "generated_epsilon{}.jpg".format(epsilon_attack)
                else:
                    file_name = str(epoch) + "generated_gan.jpg"
                visualize_images(
                    generated_adv,
                    os.path.join(
                        self.adversarial_dir,
                        file_name
                    )
                )

            self.eval(partial=10, use_val=self.params['use_val'])
            self.save()
            start_batch_id = 0

    def eval(self, partial=None, use_val=True, write=True, purificator=None, evaluator=None):
        '''
        Evaluates current model
        partial:(int) if partial is None, evaluate on partial batches
        mode: evaluates on val or test set.
        '''
        testX = self.test_X if use_val else self.train_X
        testy = self.test_y if use_val else self.train_y
        iters = len(testX) // self.batch_size
        mistakes_indices = []
        new_data = []
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
            if purificator is not None:
                old_batch_data = batch_data
                # batch_bpd = evaluator.get_bpd(batch_data)
                # print(np.mean(batch_bpd))
                batch_data = purificator.purify(batch_data)
                # batch_bpd = evaluator.get_bpd(batch_data)
                # bpds.append(batch_bpd)
                new_data.append(batch_data)
                print(np.sum((old_batch_data - batch_data)**2)/len(batch_data))
                # print(np.mean(batch_bpd))
            loss_, accuracy_, preds = self.sess.run(
                [self.loss_clean, self.accuracy_clean, self.prediction_clean],
                feed_dict={self.x_clean: 0.5 * (batch_data + 1), self.label_clean: batch_label, self.is_training: False})
            total_loss += loss_
            total_accuracy += accuracy_
            mistakes = (batch_label != preds).nonzero()[0] + i * self.batch_size
            mistakes_indices.append(mistakes)
        total_loss /= iters
        total_accuracy /= iters
        mistakes_indices = np.concatenate(mistakes_indices)
        if write:
            test_summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=total_accuracy),
                tf.Summary.Value(tag='Loss', simple_value=total_loss)])
            self.test_writer.add_summary(test_summary, self.counter)
            self.test_writer.flush()
        else:
            if use_val:
                name = "val"
            else:
                name = "train"
            np.save(os.path.join(self.checkpoint_dir, name + '_mistakes.npy'), mistakes_indices)
            if purificator is not None:
                np.save(os.path.join(self.checkpoint_dir, name + '_images_purified_{}_{}.npy'.format(
                    generator.params['epsilon'],
                    generator.params['mode']
                )), np.concatenate(new_data))
        print("[*] Evaluated model accuracy {}, loss {}".format(total_accuracy, total_loss))

    def generate_adversarial_examples(self, num=64):
        '''
        Generates FGSM Adversarial examples
        '''
        batch_data = self.train_X[:num]
        batch_label = self.train_y[:num]
        result_images, result_logits, original_logits = self.sess.run(
            [self.x_adv, self.logits_adv, self.original_logits], feed_dict={
                self.input_adv: 0.5 * (batch_data + 1),
                self.label_adv: batch_label,
                self.is_training: False,
                self.epsilon_attack: self.params['epsilon']
            }
        )
        return result_images, result_logits, original_logits, batch_data, batch_label

    def evaluate_adversarial_robustness(self, num=64):
        '''
        Evaluates accuracy of classifier with adversarial examples
        '''
        if self.params['adversarial_mode'] == "fgsm":
            self.build_graph_adversarial()
        images, logits, original_logits, original_images, original_labels = \
            self.generate_adversarial_examples(num)
        visualize_images(
            original_images + 1,
            path = os.path.join(self.adversarial_dir, "original_images.jpg"))
        visualize_images(
            images,
            path = os.path.join(self.adversarial_dir, "generated_images_epsilon{}.jpg".format(self.params['epsilon'])))
        original_accuracy = np.mean(original_labels == np.argmax(original_logits, axis=-1))
        new_accuracy = np.mean(original_labels == np.argmax(logits, axis=-1))
        print("[*] Original accuracy {}, new accuracy after attack {}".format(
            original_accuracy, new_accuracy))

    def get_bpd_dist(self, gmm_model, indices=None, use_val=True):
        '''
        Evaluates distribution bpd of misclassified and properlyclassifier examples
        using a given likelihood model
        gmm_model: PixelTrainer object with loaded checkpoint
        indices (opt): indices to consider
        use_val: use test or train set
        '''
        testX = self.test_X if use_val else self.train_X
        testy = self.test_y if use_val else self.train_y
        if indices is not None:
            testX = testX[indices]
            testy = testy[indices]
            print(len(indices), len(testy))
        iters = len(testX) // self.batch_size

        bpds = []
        is_correct = []
        true_samples = 0
        false_samples = 0
        for i in range(iters):
            batch_data = testX[i * self.batch_size:(i + 1) * self.batch_size]
            batch_label = testy[i * self.batch_size:(i + 1) * self.batch_size]
            predictions = self.sess.run(
                self.prediction_clean,
                feed_dict={self.x_clean: 0.5 * (batch_data + 1), self.label_clean: batch_label, self.is_training: False})
            true_samples += np.sum(batch_label == predictions)
            false_samples += np.sum(batch_label != predictions)
            assert (np.sum(batch_label == predictions) + np.sum(batch_label != predictions) == len(batch_label))
            bpds.append(gmm_model.get_bpd(batch_data))
            is_correct.append(batch_label == predictions)
        if iters * self.batch_size < len(testX):
            batch_data = testX[iters * self.batch_size:]
            batch_label = testy[iters * self.batch_size:]
            predictions = self.sess.run(
                self.prediction_clean,
                feed_dict={self.x_clean: 0.5 * (batch_data + 1), self.label_clean: batch_label, self.is_training: False})
            true_samples += np.sum(batch_label == predictions)
            false_samples += np.sum(batch_label != predictions)
            assert (np.sum(batch_label == predictions) + np.sum(batch_label != predictions) == len(batch_label))
            bpds.append(gmm_model.get_bpd(batch_data))
            is_correct.append(batch_label == predictions)
        print("Number of true samples, false samples and accuracy", true_samples, false_samples, true_samples/(false_samples + true_samples + 1e-12))
        bpds = np.concatenate(bpds)
        is_correct = np.concatenate(is_correct)
        return bpds, is_correct
