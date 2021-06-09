import unittest
import numpy as np
from utils import greedy_core_set
import torch
from modules import Discriminator, Generator, GaussianMixture
from utils import compute_gradient_penalty
from copy import deepcopy


class GCSTest(unittest.TestCase):
    def test_subset(self):
        x = torch.tensor(np.random.rand(100, 10))
        x_core = greedy_core_set(x, 32)
        for core_element in x_core:
            self.assertTrue(core_element in x)

    def test_functionality(self):
        x = torch.tensor(np.array([[0, 0],
                                   [20, 19],
                                   [19, 20],
                                   [21, 20],
                                   [20, 21],
                                   [100, 100]]))
        self.assertTrue(torch.equal(greedy_core_set(x, 1), x[[5]]))
        self.assertTrue(torch.equal(greedy_core_set(x, 2).sort(dim=0)[0], x[[0, 5]].sort(dim=0)[0]))

    def test_small(self):
        x = torch.tensor(np.random.rand(30, 10))
        self.assertEqual(len(greedy_core_set(x, 32)), len(x))


class GANTest(unittest.TestCase):
    def test_gan(self):
        # models settings
        lambda_gp = 0.1
        Tensor = torch.FloatTensor
        generator = Generator(2, 2)
        discriminator = Discriminator(2)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer_D.zero_grad()

        # run models for test batch
        real_data = Tensor(np.random.normal(0, 1, (64, 2)))
        z = Tensor(np.random.normal(0, 1, (64, 2)))
        real_target = Tensor(real_data.size(0), 1).fill_(1.0)
        fake_target = Tensor(real_data.size(0), 1).fill_(0.0)

        g_before = deepcopy(generator)
        d_before = deepcopy(discriminator)

        # Generate a batch of images
        fake_data = generator(z)

        # Real images
        real_validity = discriminator(real_data)
        # Fake images
        fake_validity = discriminator(fake_data)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data, Tensor)

        # Discriminator loss
        d_loss = criterion(real_validity, real_target) \
                 + criterion(fake_validity, fake_target) \
                 + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # Assert that D changed and G not changed
        g_changed = [torch.equal(after, before) for after, before in
                     zip(generator.parameters(), g_before.parameters())]
        self.assertTrue(all(g_changed))

        d_changed = [torch.equal(after, before) for after, before in
                     zip(discriminator.parameters(), d_before.parameters())]
        self.assertFalse(all(d_changed))
        optimizer_G.zero_grad()

        # Train on fake samples

        g_before = deepcopy(generator)
        d_before = deepcopy(discriminator)

        fake_data = generator(z)
        fake_validity = discriminator(fake_data)
        g_loss = criterion(fake_validity, real_target)
        g_loss.backward()
        optimizer_G.step()

        # Assert that G changed and D not changed
        g_changed = [torch.equal(after, before) for after, before in
                     zip(generator.parameters(), g_before.parameters())]
        self.assertFalse(all(g_changed))

        d_changed = [torch.equal(after, before) for after, before in
                     zip(discriminator.parameters(), d_before.parameters())]
        self.assertTrue(all(d_changed))


class GausTest(unittest.TestCase):
    def test_metrics(self):
        # Points from one cluste
        gaus = GaussianMixture(6)
        rm_value, hqs_value = gaus.evaluate(np.ones((100, 2)))
        self.assertEqual(rm_value, 1 / (6 ** 2))
        self.assertEqual(hqs_value, 1)

        # Perfect points
        grid_side = np.linspace(-(6 - 1) * 2 / 2, (6 - 1) * 2 / 2, 6)
        gaus_means = np.array([(i, j) for i in grid_side for j in grid_side])
        rm_value, hqs_value = gaus.evaluate(gaus_means)
        self.assertEqual(rm_value, 1)
        self.assertEqual(hqs_value, 1)

        # Perfect points with noise from acceptable range
        gaus_means += (np.random.rand(6 ** 2, 2) - 0.5) / 10
        rm_value, hqs_value = gaus.evaluate(gaus_means)
        self.assertEqual(rm_value, 1)
        self.assertEqual(hqs_value, 1)


if __name__ == '__main__':
    unittest.main()
