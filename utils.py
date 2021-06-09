import pandas as pd
import io
import PIL
import numpy as np
import torch
from torchvision.transforms import ToTensor
import torch.autograd as autograd
from scipy.spatial.distance import cdist


def plot_to_tensor(fig_plot):
    buf = io.BytesIO()
    fig_plot.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def greedy_core_set(x, k, return_index=False):
    '''
    Implementation of GreedyCoreSet algorithm from paper

    :param torch.tensor x: full minibatch
    :param int k: size of subset
    :param bool return_index: return samples or indexes(if use embeddings)
    :return: torch.tensor
    '''
    # Return full batch if k >= |x|
    if x.shape[0] < (k + 1):
        return list(np.arange(x.shape[0])) if return_index else x

    # For saving calculated distances
    distances = cdist(x.cpu().numpy(), x.cpu().numpy())
    np.fill_diagonal(distances, np.inf)
    other_indexes = list(np.arange(x.shape[0]))

    # First element - furthest from all others
    first_point = distances.min(axis=0).argmax()
    subset_indexes = [first_point]
    other_indexes.remove(first_point)

    # Adding other elements
    while len(subset_indexes) < k:
        new_point = distances[subset_indexes, :][:, other_indexes].min(axis=0).argmax()
        new_point = other_indexes[new_point]
        subset_indexes.append(new_point)
        other_indexes.remove(new_point)

    return subset_indexes if return_index else x[subset_indexes]



def compute_gradient_penalty(discriminator, real_samples, fake_samples, tensor_type):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = tensor_type(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = tensor_type(real_samples.shape[0], 1).fill_(1.0)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
