import torch
import torch.nn as nn
from src.base import Attack
from src.utils.functions import rescale_adversarial


class RandomNoise(Attack):
    '''
    Random noise used for comparison.

    Extra Args:
        eps (float): value used to rescale the sign tensor. If not given, searches the
            minimum epsilon that manages to fool the network.
        try_eps_min (float): minimum epsilon tested when `eps` is not given.
        try_eps_max (float): maximum epsilon tested when `eps` is not given.
        try_eps_step (float): step used to increment the tested epsilon between
            `try_eps_min` and `try_eps_max` when `eps` is not given.
    '''
    def __init__(self, model, datasource, eps=0.2, max_2_norm=None, dissimilarity=None, **kwargs):
        super().__init__(model, datasource, **kwargs)
        self.eps = eps
        self.max_2_norm = max_2_norm
        self.dissimilarity = dissimilarity

    def __call__(self, images, labels):
        # Generates some random noise
        noise = torch.zeros(images.size()).uniform_(-1, 1).to(self.datasource.device)

        # Rescales the noise
        if self.max_2_norm or self.dissimilarity:
            u_images = torch.stack([self.datasource.unnormalize(i) for i in images])
            norm_2_clean = torch.norm(torch.flatten(u_images, start_dim=1), dim=1)

            if self.max_2_norm:
                adv = rescale_adversarial(
                    images + noise, u_images, norm_2_clean, self.datasource, norm=self.max_2_norm)

            elif self.dissimilarity:
                adv = rescale_adversarial(
                    images + noise, u_images, norm_2_clean, self.datasource, self.dissimilarity)

        elif self.eps:
            adv = images + (noise * self.eps)

        # Clamps the images with the noise to the dataset range
        return self.datasource.clamp(adv)
