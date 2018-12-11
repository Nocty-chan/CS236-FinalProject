import argparse
from utils import *
from trainers.gan_trainer import GAN_CIFAR
import pprint
import json

parser = argparse.ArgumentParser("GAN Training")
# Logging options
parser.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name to save logs and checkpoints under.')
parser.add_argument('--log_freq', type=int, default=100, help='Frequency of logs during training within an epoch. -1 if no logs')
parser.add_argument('--mode', type=str, default="train", help ='Mode train |eval')

# General training parameters
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--dropout', type=float, default=0.5, help="dropout rate")
parser.add_argument('--max_iters', type=int, default=100000, help="number of max iters")
parser.add_argument('--use_val', action="store_true", help='use validation set')
parser.add_argument('--lr', type=float, default=2e-4, help="learning rate for GAN training")
parser.add_argument('--dataset', type=str, default="cifar100", help="cifar100 | cifar10 | mnist")

# GAN training
parser.add_argument('--z_dim', type=int, default=256, help="[GAN] Latent variable dimension")
parser.add_argument('--disc_iters', type=int, default=5, help="[GAN] Iterations for discriminator")
parser.add_argument('--beta1', type=float, default=0, help="[GAN] Beta1 for Adam")
parser.add_argument('--beta2', type=float, default=0.9, help="[GAN] Beta2 for Adam")
parser.add_argument('--dim_D', type=int, default=256, help="[GAN] Discriminator dimension 32 for mnist 128 for cifar")
parser.add_argument('--dim_G', type=int, default=256, help="[GAN] Generator dimension")
args = parser.parse_args()

experiment_name = args.experiment_name
if experiment_name is None:
        experiment_name = "GAN_bs{}_lr{}_z{}_dim{}_dp{}".format(args.batch_size, args.lr, args.z_dim, args.dim_D, args.dropout)

params = {
    "batch_size": args.batch_size,
    "log_dir": "logs/GAN/",
    "use_val": False,
    "flatten": False,
    "get_int": False,
    "dataset": args.dataset,
    "dropout": args.dropout,
    "gen_bs_multiple": 2,
    "max_iters": args.max_iters,
    "dim_D": args.dim_D,
    "dim_G": args.dim_G,
    "output_dim": 3072,
    "img_size": 32,
    "img_channel": 3,
    "lr": args.lr,
    "n_critic": 5,
    "acgan_scale": 1,
    "acgan_scale_G": 0.1,
    "experiment_name": experiment_name,
    "num_classes": 10 if args.dataset == "cifar10" else 100,
    "z_dim": args.z_dim
}

pprint.PrettyPrinter(indent=4).pprint(params)

with tf.Session() as sess:
    train_wgan = GAN_CIFAR(params, sess, name="GAN")
    if args.mode == "train":
        train_wgan.train()
    elif args.mode == "eval":
        train_wgan.load_checkpoint()
        train_wgan.interpolate_class("classes")
        train_wgan.generate_image("final")
        train_wgan.visualize_results(name="final", num=100, y=np.arange(100))
