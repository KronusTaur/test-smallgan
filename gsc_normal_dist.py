import argparse
import numpy as np


import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import torch

from modules import Discriminator, Generator, GaussianMixture
from utils import plot_to_tensor, greedy_core_set
from torch.utils.tensorboard import SummaryWriter


def create_distplot(array_1, array_2, name):
    plt.figure()
    figure = sns.distplot(array_1, kde=False, bins=20, norm_hist=True)
    figure = sns.distplot(array_2, kde=False, bins=20, norm_hist=True)
    figure.legend(['Original', 'Selected'])
    image = plot_to_tensor(figure.figure)
    writer.add_image(name, image)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--prior_factor", type=int, default=4, help="Over-sampling factor for prior distribution p(z)")
parser.add_argument("--target_factor", type=int, default=8, help="Over-sampling factor for target distribution p(x)")
parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--grid_side_size", type=int, default=9, help="Points count by side in 2D gaussians grid")


opt = parser.parse_args()

# Write learning info
writer = SummaryWriter('runs/2d_gaussain_mixture')
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
Tensor = torch.FloatTensor

# Initialize generator and discriminator
gaussian = GaussianMixture(opt.grid_side_size)

# Train dataset
train_samples = gaussian.sample(100000)
train_dataloader = torch.utils.data.DataLoader(
    train_samples,
    opt.batch_size * opt.target_factor,
    shuffle=True,
)

real_samples = np.zeros((0, 2))
selected_samples = np.zeros((0, 2))


real_samples_prior = np.zeros((0, 2))
selected_samples_prior = np.zeros((0, 2))

for i, batch in enumerate(train_dataloader):
    # Target samples
    data = batch.type(Tensor)
    real_samples = np.append(real_samples, data.numpy(), axis=0)
    selected_data = greedy_core_set(data, opt.batch_size).numpy()
    selected_samples = np.append(selected_samples, selected_data, axis=0)

    # Prior samples
    z = Tensor(np.random.normal(0, 1, (opt.batch_size * opt.target_factor, 2)))
    real_samples_prior = np.append(real_samples_prior, z.numpy(), axis=0)
    selected_z = greedy_core_set(z, opt.batch_size).numpy()
    selected_samples_prior = np.append(selected_samples_prior, selected_z, axis=0)

# Save distribution images
create_distplot(real_samples[:, 0], selected_samples[:, 0], 'Target, x axis, orig vs selected')
create_distplot(real_samples[:, 1], selected_samples[:, 1], 'Target, y axis, orig vs selected')
create_distplot(real_samples_prior[:, 0], selected_samples_prior[:, 0], 'Prior, x axis, orig vs selected')
create_distplot(real_samples_prior[:, 1], selected_samples_prior[:, 1], 'Prior, y axis, orig vs selected')
