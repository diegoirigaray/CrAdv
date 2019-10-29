import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from src.base import DataSource
from .imagenet_classes import imagenet_classes


# R -> -2.118 a 2.249
# G -> -2.036 a 2.429
# B -> -1.804 a 2.640
# (x / 255) * 4.4253

class ImageNetInception3(DataSource):
    def __init__(self, **kwargs):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        super().__init__(mean, std, 'imagenet', **kwargs)

        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.classes = imagenet_classes

    def get_dataset(self, train):
        return ImageFolder(os.path.join(self.path, 'train' if train else 'val'),
                           self.transform)

    def get_classes(self):
        return self.classes
