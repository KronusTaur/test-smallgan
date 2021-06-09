import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules import Discriminator, Generator, GaussianMixture
from utils import plot_to_tensor, greedy_core_set, compute_gradient_penalty
from torch.utils.tensorboard import SummaryWriter


# Parse settings
parser = argparse.ArgumentParser()
parser.add_argument("--learning_type", type=str, default="gan", help="Type of learning - smallgan or wgan-gp")
parser.add_argument("--train_steps", type=int, default=120000, help="number of standard steps of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--valid_size", type=int, default=10000, help="Validation dataset size")
parser.add_argument("--valid_steps", type=int, default=100, help="Interval between validations in epochs")
parser.add_argument("--valid_batch_size", type=int, default=512, help="Validation dataset size")
parser.add_argument("--prior_factor", type=int, default=4, help="Over-sampling factor for prior distribution p(z)")
parser.add_argument("--target_factor", type=int, default=8, help="Over-sampling factor for target distribution p(x)")
parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--grid_side_size", type=int, default=8, help="Points count by side in 2D gaussians grid")
parser.add_argument("--image_save_step", type=int, default=10000, help="Writing validation prior and generation images")
parser.add_argument("--lambda_gp", type=int, default=0.1, help="Weight for gradient penalty in loss")

parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="Batch generation cpu threads")
parser.add_argument("--cuda", type=bool, default=True, help="Using GPU if available")
parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
if __name__ == '__main__':
    opt = parser.parse_args()

    # Write learning info
    writer = SummaryWriter('runs/2d_gaussain_mixture')
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    cuda = True if torch.cuda.is_available() and opt.cuda else False
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() and opt.cuda else torch.FloatTensor

    # Loss weight for gradient penalty
    lambda_gp = opt.lambda_gp

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, 2)
    discriminator = Discriminator(2)
    gaussian = GaussianMixture(opt.grid_side_size)
    train_data = gaussian.sample(opt.batch_size * opt.target_factor if opt.learning_type == 'smallgan' else opt.batch_size)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    prior_batch_size = opt.batch_size * opt.prior_factor if opt.learning_type == 'smallgan' else opt.batch_size

    # Valid dataset
    valid_samples = np.random.uniform(-1, 1, (opt.valid_size, opt.latent_dim))
    valid_dataloader = torch.utils.data.DataLoader(
        valid_samples,
        batch_size=opt.valid_batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    criterion = nn.BCEWithLogitsLoss()

    # Train
    for global_step in range(opt.train_steps):
        # Configure input and apply GreedyCoreset
        real_data = Tensor(next(train_data))
        if opt.learning_type == 'smallgan':
            real_data = greedy_core_set(real_data, opt.batch_size)
        optimizer_D.zero_grad()

        # Sample noise as generator input and apply GreedyCoreset
        z = Tensor(np.random.uniform(-1, 1, (prior_batch_size, opt.latent_dim)))
        if opt.learning_type == 'smallgan':
            z = greedy_core_set(z, opt.batch_size)

        # Generate a batch of images
        fake_data = generator(z)

        # Real and fake data
        real_validity = discriminator(real_data)
        fake_validity = discriminator(fake_data)

        # Ground truths
        real = Tensor(real_validity.size(0), 1).fill_(1.0)
        fake = Tensor(fake_validity.size(0), 1).fill_(0.0)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data, Tensor)

        # Discriminator loss
        d_loss = criterion(real_validity, real) + criterion(fake_validity, fake) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if global_step % opt.n_critic == 0:
            fake_data = generator(z)
            fake_validity = discriminator(fake_data)
            # Generator loss
            g_loss = criterion(fake_validity, real)
            g_loss.backward()
            optimizer_G.step()

        # Validation
        if global_step % opt.valid_steps == 0:
            generator.eval()
            with torch.no_grad():
                gen_results = np.zeros((0, 2))
                for j, val_batch in enumerate(valid_dataloader):
                    val_batch = val_batch.type(Tensor)
                    gen_batch = generator(val_batch)
                    gen_results = np.append(gen_results, gen_batch.detach().cpu().numpy(), axis=0)
                recovered_modes_value, hq_samples_value = gaussian.evaluate(gen_results, 4)
                if global_step % (opt.valid_steps * 50) == 0:
                    print('Steps %d, validation:  [Recovered modes: %f]  [High quality samples: %f]'
                        % (global_step, recovered_modes_value * 100, hq_samples_value * 100))
                writer.add_scalars('recovered_modes/{}_modes'.format(opt.grid_side_size ** 2),
                                {'{}_type'.format(opt.learning_type): recovered_modes_value},
                                global_step)
                writer.add_scalars('hq_samples/{}_modes'.format(opt.grid_side_size ** 2),
                                {'{}_type'.format(opt.learning_type): hq_samples_value},
                                global_step)

        # Save images
        if global_step % opt.image_save_step == 0:
            # Validation calculate
            generator.eval()
            gen_results = np.zeros((0, 2))
            with torch.no_grad():
                for j, val_batch in enumerate(valid_dataloader):
                    val_batch = val_batch.type(Tensor)
                    gen_batch = generator(val_batch)
                    gen_results = np.append(gen_results, gen_batch.detach().cpu().numpy(), axis=0)

            # Get modes
            _, _, modes = gaussian.evaluate(gen_results, 4, get_modes=True)

            # Create images
            df_prior = pd.DataFrame(
                valid_samples,
                columns=['x', 'y'],
            )
            df_prior['mode'] = modes
            df_target = pd.DataFrame(
                gen_results,
                columns=['x', 'y'],
            )
            plt.figure()
            prior_plot = sns.pairplot(df_prior, x_vars='x', y_vars='y', hue='mode', height=5)
            prior_image = plot_to_tensor(prior_plot)
            writer.add_image('{} modes, {}, Prior by mode'.format(opt.grid_side_size ** 2, opt.learning_type),
                            prior_image, global_step)
            plt.close()

            plt.figure()
            target_plot = sns.pairplot(df_target, x_vars='x', y_vars='y', height=5)
            target_image = plot_to_tensor(target_plot)
            writer.add_image('{} modes, {}, target values, 2D'.format(opt.grid_side_size ** 2, opt.learning_type),
                            target_image, global_step)
            plt.close()

            plt.figure()
            target_1d_x_plot = sns.distplot(gen_results[:, 0], kde=False,
                                            bins=opt.grid_side_size * 2, norm_hist=True)
            target_1d_x_image = plot_to_tensor(target_1d_x_plot.figure)
            writer.add_image('{} modes, {}, target hist, 1D'.format(opt.grid_side_size ** 2, opt.learning_type),
                            target_1d_x_image, global_step)
            plt.close()
