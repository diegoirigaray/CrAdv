import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.base import DataSource


class CIFAR10(DataSource):
    def __init__(self, **kwargs):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        super().__init__(mean, std, 'cifar10', **kwargs)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                        'horse', 'ship', 'truck')

    def get_dataset(self, train):
        return datasets.CIFAR10(
            root=self.path, train=train, transform=self.transform, download=True)

    def get_classes(self):
        return self.classes
