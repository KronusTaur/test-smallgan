import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, dim=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, output_dim),
        )

    def forward(self, z):
        output = self.model(z)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_size, dim=512):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity


class GaussianMixture:
    def __init__(self, grid_side_size, gaus_std=0.05, cell_size=2):
        '''

        :param grid_side_size: number of grid points by side
        :param gaus_std: gaussians standard deviation, default=0.05 from papers experiments
        :param cell_size: , default=2 from papers experiments
        '''
        self.grid_side_size = grid_side_size
        self.gaus_std = gaus_std
        self.cell_size = cell_size

        self.gaus_number = grid_side_size ** 2
        self.grid_side = np.linspace(-(grid_side_size - 1) * cell_size / 2,
                                     (grid_side_size - 1) * cell_size / 2,
                                     grid_side_size)
        self.gaus_means = np.array([(i, j) for i in self.grid_side for j in self.grid_side])
        self.gaus_covar = np.eye(2) * gaus_std ** 2

    def sample(self, size):
        '''
        Sampling from gaussian distributions mixture
        '''
        while True:
            # Sampling distributions and count every type
            samples_count_by_gaus = np.bincount(np.random.randint(self.gaus_number, size=size))

            # Sampling from distributions and appending for full array
            gen_samples = np.ones((0, 2))
            for j, samples_count in enumerate(samples_count_by_gaus):
                gen_samples = np.append(gen_samples,
                                        np.random.multivariate_normal(self.gaus_means[j],
                                                                      self.gaus_covar,
                                                                      samples_count),
                                        axis=0)

            # Shuffle elements
            np.random.shuffle(gen_samples)
            yield gen_samples

    def evaluate(self, generated_samples, hq_std_number=4, get_modes=False):
        '''
        :param np.array generated_samples: generated samples
        :param int hq_std_number: number of std for high quality samples distance
        :pararm bool get_modes: return modes
        :return: tuple (recovered_modes_percent, high_quality_percent)
        '''
        # L1 pairwise matrix
        distance_matrix = cdist(generated_samples, self.gaus_means)

        # Mins and argmins
        mins = distance_matrix.min(axis=1)
        argmins = distance_matrix.argmin(axis=1)

        # Calculate metrics
        high_qual = mins < hq_std_number * self.gaus_std
        high_quality = high_qual.mean()
        recovered_modes = len(np.unique(argmins[high_qual])) / self.gaus_number
        if get_modes:
            return recovered_modes, high_quality, argmins
        else:
            return recovered_modes, high_quality
