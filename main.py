import argparse
import scipy.misc
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pprint
import numpy as np

from models.densenet import dense_net
from trainers.classifier_trainer import SGDTrainer
from trainers.gan_trainer import GAN_CIFAR
from trainers.attack_trainer import AttackTrainer
from utils import visualize_images, check_folder, visualize_histogram

parser = argparse.ArgumentParser("Adversarial Training")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--dropout', type=float, default=0.5, help="dropout rate")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--mode', type=str, default='train', help='train | eval | bdp_eval | gen_adversarial')
parser.add_argument('--use_val', action="store_true", help='use validation set')
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate for classifier training")
parser.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name to save logs and checkpoints under.')
parser.add_argument('--log_freq', type=int, default=100, help='Frequency of logs during training within an epoch. -1 if no logs')

# L2 Regularizer
parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 Regularization strength")

# DenseNet options
parser.add_argument('--growth_rate', type=int, default=12, help="[DenseNet] Growth Rate")
parser.add_argument('--depth', type=int, default=100, help="[DenseNet] depth of Dense layers")
parser.add_argument('--theta', type=float, default=0.5, help="[DenseNet] compression rate")

# Adversarial training option
parser.add_argument('--adversarial_mode', type=str, default=None, help="Specify type of adversarial agumentation [None | fgsm | gan")
parser.add_argument('--prop', type=float, default=0.5, help="[Adversarial] Proportion of adversarial examples to compute in batch")
parser.add_argument('--adv_weight', type=float, default=0.3, help="[Adversarial] Controls relative weight of loss on adversarial examples")
parser.add_argument('--gen_name', type=str, help="[Adversarial] Name of generative model to load")

# adversarial options
parser.add_argument('--epsilon', type=float, default=0.1, help="[Adversarial] Attack epsilon")

# GAN options
parser.add_argument('--z_dim', type=int, default=256, help="[GAN] Latent variable dimension")
parser.add_argument('--disc_iters', type=int, default=5, help="[GAN] Iterations for discriminator")
parser.add_argument('--beta1', type=float, default=0, help="[GAN] Beta1 for Adam")
parser.add_argument('--beta2', type=float, default=0.9, help="[GAN] Beta2 for Adam")
parser.add_argument('--dim_D', type=int, default=256, help="[GAN] Discriminator dimension 32 for mnist 128 for cifar")
parser.add_argument('--dim_G', type=int, default=256, help="[GAN] Generator dimension")

# Adversarial generative examples options
parser.add_argument('--lambda1', type=float, default=0.1, help="[AC-GAN-ADV] lambda1 factor")
parser.add_argument('--lambda2', type=float, default=0.1, help="[AC-GAN-ADV] lambda2 factor")
parser.add_argument('--z_eps', type=float, default=0.1, help="[AC-GAN-ADV] z_epsilon")
parser.add_argument('--adv_lr', type=float, default=0.1, help="[AC-GAN-ADV] learning rate")
parser.add_argument('--num_iters', type=int, default=1000, help="[AC-GAN-ADV] num iterations")

args = parser.parse_args()

experiment_name = args.experiment_name
if experiment_name is None:
    experiment_name = "lr_{}_dp_{}_bs_{}_L{}_k{}_theta{}".format(
        args.lr, args.dropout, args.batch_size, args.depth, args.growth_rate, args.theta)
    if args.adversarial_mode is not None:
        experiment_name = "adversarial_{}_".format(args.adversarial_mode) + experiment_name + "_adv{}_prop{}".format(args.adv_weight, args.prop)

params = {
    # Utility parameters
    'log_dir': 'logs/densenet',
    'experiment_name': experiment_name,
    'log_freq': args.log_freq,
    'model_name': 'densenet',
    # Dataset relate parameters
    'dataset': 'cifar100',
    'num_classes': 100,
    'img_size': 32,
    'img_channel': 3,
    'flatten': False,
    # General training parameters
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'optimizer': "Momentum",
    'lr': args.lr,
    'use_val': args.use_val,
    'flatten': False,
    'get_int': False,
    # Regularization
    'weight_decay': args.weight_decay,
    'dropout': args.dropout, # should be 0.2 for DenseNet
    # DenseNet parameters
    'growth_rate': args.growth_rate,
    'depth': args.depth,
    'theta': args.theta,
    # Adversarial
    'adversarial_mode': args.adversarial_mode,
    'prop': args.prop,
    'lambda': args.adv_weight,
    # FGSM attack options
    'epsilon': args.epsilon,
}

gan_params = {
    "log_dir": "logs/GAN/",
    "dataset": "cifar100",
    "use_val": False,
    "flatten": False,
    "get_int": False,
    "dropout": args.dropout,
    "img_size": 32,
    "img_channel": 3,
    "batch_size": args.batch_size,
    "gen_bs_multiple": 2,
    "iters": None,
    "dim_D": args.dim_D,
    "dim_G": args.dim_G,
    "output_dim": 3072,
    "lr": args.lr,
    "max_iters": 100000,
    "n_critic": 5,
    "acgan_scale": 1,
    "acgan_scale_G": 0.1,
    "experiment_name": args.gen_name,
    "num_classes": 100,
    "z_dim": args.z_dim,
    "dataset_name": "cifar100"
}

pprint.PrettyPrinter(indent=4).pprint(params)

with tf.Session() as sess:
    net = tf.make_template('classifier', dense_net)
    if args.adversarial_mode=="gan":
        gen = GAN_CIFAR(gan_params, sess, mode="eval", name="GAN")
    else:
        gen = None
    train_cifar = SGDTrainer(sess, net, gen, params)
    if args.mode == 'train':
        train_cifar.train()
        train_cifar.eval(use_val=False, write=False)
    if args.mode == 'eval':
        train_cifar.load_checkpoint()
        train_cifar.eval(use_val=True, write=False)
        train_cifar.eval(use_val=False, write=False)
    elif args.mode == 'gen_adversarial':
        if args.adversarial_mode == "gan":
            attack_params = {
                "num": 100,
                "epsilon": args.epsilon,
                "z_eps": args.z_eps,
                "lambda1": args.lambda1,
                "lambda2": args.lambda2,
                "lr": args.adv_lr,
                "log_dir": "logs/",
                "num_classes": 100,
            }
            attack_trainer = AttackTrainer(sess, attack_params, gen, train_cifar)
            source_class = [i for i in range(100)]
            target_class = [((i + 1) % 100) for i in range(100)]
            samples, original = attack_trainer.generate(source_class, target_class, args.num_iters)
            file_name = "gan_adversarial_succesful_samples.jpg"
            samples = np.stack(samples)
            visualize_images(
                0.5 * (samples + 1),
                path=os.path.join(
                    attack_trainer.images_dir,
                    file_name
                )
            )
            original = np.stack(original)
            visualize_images(
                0.5 * (original + 1),
                path = os.path.join(
                    attack_trainer.images_dir,
                    "gan_adversarial_succesful_original.jpg"
                )
            )
            closest = attack_trainer.get_closest(samples)
            visualize_images(
                0.5 * (closest + 1),
                path = os.path.join(attack_trainer.images_dir,
                                    "gan_adversarial_closest.jpg")
            )
        else:
            train_cifar.load_checkpoint()
            train_cifar.evaluate_adversarial_robustness(num=256)
