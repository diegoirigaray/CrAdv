import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.base import DataSource


class FashionMNIST(DataSource):
    def __init__(self, **kwargs):
        mean = [0.5]
        std = [0.5]
        super().__init__(mean, std, 'fmnist', **kwargs)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                        'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    def get_dataset(self, train):
        return datasets.FashionMNIST(
            root=self.path, train=train, transform=self.transform, download=True)

    def get_classes(self):
        return self.classes
