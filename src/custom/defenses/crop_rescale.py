import torch
from torch.nn import functional as F
from random import randint
from torchvision import transforms
from src.base import Defense


class CropRescale(Defense):
    '''
    Basic implementation of the `croping and rescaling` defense.

    Reference:
        Countering adversarial images using input transformations.
        https://arxiv.org/pdf/1711.00117.pdf

    Extra Args:
        crop_size (int): size of the cropped square.
        crop_size_factor (float): if `crop_size` is not given, the used crop size is
            obtained by multiplaying the first dataset sampe's size by this factor.
        num_samples (int): number of cropped samples used to obtain an avarage result.
    '''
    def __init__(self, model, datasource, crop_size=None, crop_size_factor=0.75,
                 num_samples=10, **kwargs):
        super().__init__(model, datasource, **kwargs)
        if not crop_size:
            if not crop_size_factor or crop_size_factor > 1:
                raise TypeError('Invalid `crop_size_factor`.')
            sample = datasource.get_dataset(False)[0][0]
            crop_size = int(min(sample.size(1), sample.size(2)) * crop_size_factor)
        self.crop_size = crop_size
        self.crop_size_factor = crop_size_factor
        self.num_samples = num_samples

    def crop_and_resize(self, images):
        size = images.size()[2:]
        postimgs = []
        for img in images:
            # Randomly selects the corner from where to crop the image
            rand_x = randint(0, size[0] - self.crop_size)
            rand_y = randint(0, size[1] - self.crop_size)
            new = img[:,rand_x:rand_x + self.crop_size,rand_y:rand_y + self.crop_size]
            postimgs.append(new)

        # Rescale the images
        postimgs = torch.stack(postimgs)
        postimgs = F.interpolate(postimgs, size, mode='bilinear', align_corners=False)
        return postimgs

    def forward(self, input):
        partial_results = []

        # Stores the `num_samples` predictions of each image
        for i in range(self.num_samples):
            c_r_images = self.crop_and_resize(input)
            partial_results.append(self.model(c_r_images))

        # Re arrenges the images predictions
        partial_results = torch.cat(partial_results, 1)
        partial_results = torch.reshape(
            partial_results, [input.size()[0], self.num_samples, -1])

        # final_results = torch.sum(partial_results, 1) / self.num_samples
        return partial_results.mean(1)
