import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.base import DataSource


class MNIST(DataSource):
    def __init__(self, **kwargs):
        mean = [0.5]
        std = [0.5]
        super().__init__(mean, std, 'mnist', **kwargs)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        self.classes = ('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                        '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')

    def get_dataset(self, train):
        return datasets.MNIST(
            root=self.path, train=train, transform=self.transform, download=True)

    def get_classes(self):
        return self.classes
