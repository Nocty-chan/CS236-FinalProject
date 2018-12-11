"""WGAN-GP ResNet for CIFAR-100"""
import os, sys
sys.path.append(os.getcwd())

import tflib.save_images
from models.resnet_gan import *
from trainers.trainer import Trainer
import time
import locale
locale.setlocale(locale.LC_ALL, '')

from utils import visualize_images, show_all_variables, check_folder
class GAN_CIFAR(Trainer):
    def __init__(self, params, session, name="GAN", mode="train"):
        self.generator = tf.make_template(name + '/generator', Generator)
        self.discriminator = tf.make_template(name + '/discriminator', Discriminator)
        Trainer.__init__(self, session, params, name=name, mode=mode)
        self.build_eval_graph()
        self.train_X = np.transpose(self.train_X, (0, 3, 1, 2))
        self.test_X = np.transpose(self.test_X, (0, 3, 1, 2))

    def train_impl(self, start_epoch, start_batch_id):
        while self.counter < self.params["max_iters"]:
            permutation = np.random.permutation(self.num_data)
            dataX = self.train_X[permutation]
            datay = self.train_y[permutation]

            for iteration in range(start_batch_id, self.iters):
                start_time = time.time()

                if self.counter > 0:
                    _ = self.sess.run(
                        [self.gen_train_op],
                        feed_dict={self._iteration: self.counter, self.is_training: False})

                for i in range(self.params["n_critic"]):
                    _data = self.train_X[(i + iteration * self.params["n_critic"]) * self.batch_size:(i + iteration * self.params["n_critic"] + 1) * self.batch_size]
                    _label = self.train_y[(i + iteration * self.params["n_critic"]) * self.batch_size:(i + iteration * self.params["n_critic"] + 1) * self.batch_size]
                    summary, _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = \
                        self.sess.run(
                            [
                                self.summary,
                                self.disc_cost,
                                self.disc_wgan,
                                self.disc_acgan,
                                self.disc_acgan_acc,
                                self.disc_acgan_fake_acc,
                                self.disc_train_op
                            ],
                            feed_dict={
                                self.all_real_data: _data,
                                self.all_real_labels: _label,
                                self._iteration: self.counter,
                                self.is_training: True,
                            }
                        )

                self.writer.add_summary(summary, self.counter)

                if self.counter % 100 == 0:
                    self.eval()
                    self.save()
                    self.generate_image(self.counter)
                self.counter += 1
                start_batch_id = 0

    def eval(self):
        dev_disc_accs = []
        num_iters = len(self.test_X) // self.batch_size

        for i in range(num_iters):
            images = self.test_X[i * self.batch_size:(i+1) * self.batch_size]
            _labels = self.test_y[i * self.batch_size:(i+1) * self.batch_size]

            _dev_disc_cost = self.sess.run(
                [self.disc_acgan_acc],
                feed_dict={self.all_real_data: images, self.all_real_labels: _labels, self.is_training:False})
            dev_disc_accs.append(_dev_disc_cost)

        eval_summary = tf.Summary(value=[
            tf.Summary.Value(tag="Classifier True Accuracy", simple_value=np.mean(dev_disc_accs))])
        self.test_writer.add_summary(eval_summary, self.counter)

    def build_graph(self):
        params = self.params

        self._iteration = tf.placeholder(tf.int32, shape=None)
        self.is_training = tf.placeholder(tf.bool, shape=(), name="training")
        self.all_real_data = tf.placeholder(tf.float32, shape=[params["batch_size"], params["img_channel"], params["img_size"], params["img_size"]])
        self.all_real_labels = tf.placeholder(tf.int32, shape=[params["batch_size"]])

        fake_data = self.generator(params["batch_size"], self.all_real_labels, self.params)

        all_real_data = tf.reshape(
            self.all_real_data,
            [params["batch_size"], params["output_dim"]]
        )
        all_real_data += tf.random_uniform(
            shape=[params["batch_size"], params["output_dim"]],
            minval=0.,
            maxval=1./128
        ) # dequantize

        real_and_fake_data = tf.concat([
            all_real_data,
            fake_data
        ], axis=0)
        real_and_fake_labels = tf.concat([
            self.all_real_labels,
            self.all_real_labels
        ], axis=0)
        disc_all, disc_all_acgan = self.discriminator(real_and_fake_data, real_and_fake_labels, self.params, self.is_training)
        disc_real = disc_all[:params["batch_size"]]
        disc_fake = disc_all[params["batch_size"]:]
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        disc_acgan_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=disc_all_acgan[:params["batch_size"]],
                labels=real_and_fake_labels[:params["batch_size"]])
        )
        disc_acgan_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(disc_all_acgan[:params["batch_size"]], dimension=1)),
                    real_and_fake_labels[:params["batch_size"]]
                ),
                tf.float32
            )
        )
        disc_acgan_fake_acc =tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(disc_all_acgan[params["batch_size"]:], dimension=1)),
                    real_and_fake_labels[params["batch_size"]:]
                ),
                tf.float32
            )
        )


        alpha = tf.random_uniform(
            shape=[params["batch_size"], 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - all_real_data
        interpolates = all_real_data + (alpha * differences)
        gradients = tf.gradients(
            self.discriminator(interpolates, self.all_real_labels, self.params, self.is_training)[0], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 10 * tf.reduce_mean((slopes - 1.)**2)
        disc_cost += gradient_penalty

        self.disc_wgan = disc_cost
        self.disc_acgan = disc_acgan_cost
        self.disc_acgan_acc = disc_acgan_acc
        self.disc_acgan_fake_acc = disc_acgan_fake_acc
        self.disc_cost = self.disc_wgan + (params["acgan_scale"] * self.disc_acgan)

        tf.summary.scalar("Discriminator_Loss", self.disc_cost)
        tf.summary.scalar("Discriminator WGan Loss", self.disc_wgan)
        tf.summary.scalar("Classifier loss", self.disc_acgan)
        tf.summary.scalar("Classifier Fake Accuracy", self.disc_acgan_fake_acc)
        tf.summary.scalar("Classifier True Accuracy", self.disc_acgan_acc)

        disc_params = lib.params_with_name('Discriminator.')

        decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / self.params["max_iters"]))

        n_samples = params["gen_bs_multiple"] * params["batch_size"]
        fake_labels = tf.cast(tf.random_uniform([n_samples]) * params["num_classes"], tf.int32)
        disc_fake, disc_fake_acgan = self.discriminator(
            self.generator(n_samples, fake_labels, self.params),
            fake_labels,
            self.params,
            self.is_training)
        gen_cost = -tf.reduce_mean(disc_fake)
        gen_acgan_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
        )
        gen_cost += params["acgan_scale_G"] * gen_acgan_cost
        self.gen_cost = gen_cost

        tf.summary.scalar("Generator Loss", self.gen_cost)

        self.summary = tf.summary.merge_all()
        gen_opt = tf.train.AdamOptimizer(
            learning_rate=params["lr"] * decay, beta1=0., beta2=0.9)
        disc_opt = tf.train.AdamOptimizer(
            learning_rate=params["lr"] * decay, beta1=0., beta2=0.9)
        self.gen_gv = gen_opt.compute_gradients(self.gen_cost, var_list=lib.params_with_name('Generator'))
        self.disc_gv = disc_opt.compute_gradients(self.disc_cost, var_list=disc_params)
        self.gen_train_op = gen_opt.apply_gradients(self.gen_gv)
        self.disc_train_op = disc_opt.apply_gradients(self.disc_gv)

        show_all_variables(scope="GAN")
        self.init_op = tf.global_variables_initializer()

    def build_eval_graph(self):
        # For generating samples
        self.fixed_noise = tf.constant(np.random.normal(size=(self.params['num_classes'] * 10, self.params['z_dim'])).astype('float32'))
        self.fixed_labels = tf.constant(np.array([i for i in range(self.params['num_classes'])] * 10, dtype='int32'))
        self.fixed_noise_samples = self.generator(10 * self.params['num_classes'], self.fixed_labels, self.params, noise=self.fixed_noise)

        self.noise_inference = tf.placeholder(tf.float32, shape=[None, self.params['z_dim']])
        self.labels_inference = tf.placeholder(tf.int32, shape=[None])
        self.samples_inference = self.generator(1, self.labels_inference, self.params, noise=self.noise_inference)

    def generate_image(self, name):
        samples = self.sess.run(self.fixed_noise_samples)
        self.save_image(samples, name, self.params['num_classes'])

    def visualize_results(self, name, y=None, num=100, save=True):
        z_sample = np.random.normal(size=(num, self.params['z_dim']))
        if y is None:
            y = np.random.choice(self.params['num_classes'], num)
        samples = self.sess.run(
            self.samples_inference,
            feed_dict={self.noise_inference: z_sample, self.labels_inference: y})
        if save:
            self.save_image(samples, "final_" + name, int(np.sqrt(num)))
        return samples

    def generate(self, target_classes, num):
        '''
        Input:
        target_classes: (num, ) array of classes
        num: int
        Outputs:
        samples: (num, 32, 32, 3) images
        '''
        z_sample = np.random.normal(size=(num, self.params['z_dim']))
        samples = self.sess.run(
            self.samples_inference,
            feed_dict={self.noise_inference: z_sample, self.labels_inference: target_classes})
        samples = samples.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        return samples

    def interpolate_class(self, name, num=10, save=True):
        z_sample = np.random.randn(num, self.params['z_dim'])
        y = np.arange(self.params['num_classes'])
        z_sample = np.tile(z_sample[:, np.newaxis,:], (1, self.params['num_classes'], 1))
        y = np.tile(y[np.newaxis,:], (num, 1))
        z_sample = np.reshape(z_sample,(-1, self.params['z_dim']))
        y = np.reshape(y, (-1,))
        samples = self.sess.run(
            self.samples_inference,
            feed_dict={
                self.noise_inference:z_sample,
                self.labels_inference: y
            }
        )
        if save:
            self.save_image(samples.reshape(-1, 3, 32, 32), name, self.params['num_classes'])

    def save_image(self, samples, name, num_rows):
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        images_dir = os.path.join(self.checkpoint_dir, 'samples')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        lib.save_images.save_images(
            samples.reshape((len(samples), 3, 32, 32)),
            os.path.join(images_dir, 'samples_{}.png'.format(name)),
            num_rows
        )
