import os
import torch
from src.base import Defense
from torchvision.transforms import ToPILImage, ToTensor
from torch.autograd import Function
from PIL import Image


def JPeg(image, quality):
    img = torch.clamp(image, 0, 1)
    img = ToPILImage()(img.cpu())
    path = '/tmp/def_jpeg.jpeg'
    img.save(path, 'JPEG', quality=quality)
    img = Image.open(path)
    img = ToTensor()(img)
    return img


class JpegFunction(Function):
    @staticmethod
    def forward(ctx, input, quality):
        out = torch.zeros_like(input)
        for i, img in enumerate(input):
            img = JPeg(img, quality)
            out[i] = img

        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class JpegCompression(Defense):
    '''
    Basic implementation of `JPEG compression` defense.

    Reference:
        Countering adversarial images using input transformations.
        https://arxiv.org/pdf/1711.00117.pdf

    Extra Args:
        quality (Integer): quality to use in the Jpeg compression (between 1 and 100).
    '''
    def __init__(self, model, datasource, quality=75, **kwargs):
        super().__init__(model, datasource, **kwargs)
        if quality < 1 or quality > 100:
            raise ValueError('`quality` must be in the range from 1 to 100')
        self.quality = quality

    def process(self, input):
        # Unnormalize images before using jpeg
        for img in input:
            self.datasource.unnormalize(img, True)

        out = JpegFunction.apply(input, self.quality)

        # Normalizes them again
        for img in out:
            self.datasource.normalize(img, True)
        return out
