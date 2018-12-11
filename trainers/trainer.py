import numpy as np
import os
import tensorflow as tf
from utils import show_variables

class Trainer(object):
    '''
    Generic trainer class handling logs and checkpoints and data loading (CIFAR100)
    '''
    def __init__(self, sess, params, name=None, mode="train"):
        '''
        Init:
        sess: tf.Session()
        params: dictionary of hyperparameters for the model and training.
        name: scope of variables to save, if None save all.
        mode: "train" or "eval"
        '''
        self.sess = sess
        self.params = params
        self.name = name
        self.checkpoint_dir = os.path.join(self.params['log_dir'], self.params['dataset'], self.params['experiment_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if mode != "eval":
            self.load_data(use_val=params['use_val'], flatten=params['flatten'], get_int=params['get_int'])
            self.build_graph()
        self.counter = 0
        self.initialized = False
        self.mode = mode
    def save_params(self):
        import json
        with open(self.checkpoint_dir + '/params.json', 'w') as fp:
            json.dump(self.params, fp, sort_keys=True, indent=4)
        print("[*] Parameters saved in {}".format(self.checkpoint_dir + '/params.json'))

    def build_graph(self):
        '''
        Builds computation graph for training and evaluating
        '''
        return NotImplementedError

    def load_data(self, use_val=False, flatten=False, get_int=False):
        '''
        Loads CIFAR 100 or MNIST data in val, train and test set
        use_val: if False just splits into train and test
        flatten: if True returns flattened version otherwise returns H, W, C tensor
        get_int: if False, normalizes between 0 and 1
        '''
        if self.params['dataset'] == 'mnist':
            (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
            train_x = np.reshape(train_x, (-1, 28, 28, 1))
            test_x = np.reshape(test_x, (-1, 28, 28, 1))
        elif self.params['dataset'] == 'cifar100':
            (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
        elif self.params['dataset'] == 'cifar10':
            (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        else:
            raise NotImplementedError
        if not get_int:
            self.train_X = 2 * (train_x / 255. - 0.5)
            self.test_X = 2 * (test_x / 255. - 0.5)
        else:
            self.train_X = train_x
            self.test_X = test_x

        self.train_y = np.reshape(train_y, (-1))
        self.test_y = np.reshape(test_y, (-1))
        if flatten:
            self.train_X = np.reshape(self.train_X, (-1, 32 * 32 * 3))
            self.test_X = np.reshape(self.test_X, (-1, 32 * 32 * 3))

        # #Val/Train split up
        if use_val:
            indices = np.random.RandomState(seed=42).permutation(len(self.train_X))
            num_train = int(0.9 * len(self.train_X))
            self.val_X = self.train_X[indices[num_train:]]
            self.val_y = self.train_y[indices[num_train:]]
            self.train_X = self.train_X[indices[:num_train]]
            self.train_y = self.train_y[indices[:num_train]]

        self.num_data = self.train_X.shape[0]
        self.batch_size = self.params['batch_size']
        self.iters = self.num_data // (self.batch_size * self.params.get('n_critic', 1))
        print("[*] Dataset loaded train set {}, test set {}".format(self.num_data, len(self.test_X)))


    def train_impl(self, start_epoch, start_batch_id):
        raise NotImplementedError

    def train(self):
            try:
                self.writer = tf.summary.FileWriter(self.checkpoint_dir + '/train', self.sess.graph)
                self.test_writer = tf.summary.FileWriter(self.checkpoint_dir + '/test', self.sess.graph)
                self.sess.run(self.init_op)
                # restore check-point if it exists
                start_epoch, start_batch_id = self.load_checkpoint()
                self.train_impl(start_epoch, start_batch_id)
            except KeyboardInterrupt:
                print("Interupted... saving model.")
            self.save()

    def eval(self, partial=None, use_val=True, write=True):
        raise NotImplementedError

    def save(self):
        self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_dir, self.params['experiment_name'] + '.model'),
            global_step=self.counter)
        print("[*] Saved model in {}".format(self.checkpoint_dir))

    def load(self):
        import re
        print(" [*] Reading checkpoints from {}".format(self.checkpoint_dir))

        if self.name is not None:
            print("Name is", self.name)
            variables_to_load = [v for v in tf.global_variables() if self.name in v.name]
            print("Loading and saving variables ... ")
            show_variables(variables_to_load)
            self.saver = tf.train.Saver(variables_to_load)
        else:
            print("Loading and saving variables ... ")
            show_variables(tf.global_variables())
            self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            # When no checkpoint is found, save parameters in the folder
            self.save_params()
            return False, 0

    def load_checkpoint(self):
        could_load, checkpoint_counter = self.load()
        if could_load and self.mode !='eval':
            start_epoch = (int)(checkpoint_counter / self.iters)
            start_batch_id = checkpoint_counter - start_epoch * self.iters
            self.counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            self.counter = 0
            print(" [!] Load failed...")
        return start_epoch, start_batch_id
