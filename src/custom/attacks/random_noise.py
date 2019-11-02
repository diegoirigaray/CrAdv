import torch
import torch.nn as nn
from src.base import Attack
from src.utils.functions import rescale_adversarial


class RandomNoise(Attack):
    '''
    Random noise attack used for comparison.

    Extra Args:
        eps (float): factor used to scale the random perturbation (uniform (-1,1))
        max_2_norm (float, optional): if given, the random perturbation is rescaled
            as to have this value as euclidean norm.
        dissimilarity (float, optional) if given, the random perturbation is rescaled
            as to have this value as normalized dissimilarity (euclidean norm).
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
