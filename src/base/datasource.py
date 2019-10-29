import os
import torch
from torchvision.transforms.functional import normalize
from torch.utils.data.dataloader import (DataLoader, _SingleProcessDataLoaderIter,
                                         _MultiProcessingDataLoaderIter)


class IterDeviceMixin():
    '''
    Mixin to allow DataLoaderIters to automatically move samples to device,
    avoiding the device dependecy on other modules.
    '''
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop('device')
        super().__init__(*args, **kwargs)

    def __next__(self):
        data, labels = super().__next__()
        return data.to(self.device), labels.to(self.device)


class CustomSingleIter(IterDeviceMixin, _SingleProcessDataLoaderIter):
    pass


class CustomMultiIter(IterDeviceMixin, _MultiProcessingDataLoaderIter):
    pass


class CustomDataLoader(DataLoader):
    '''
    Custom DataLoader implementation to allow 'passing' the dataset to device.
    '''
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop('device')
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self.device:
            if self.num_workers == 0:
                return CustomSingleIter(self, device=self.device)
            else:
                return CustomMultiIter(self, device=self.device)
        return super().__iter__()


class DataSource(object):
    """
    Base class for all used datasets.

    This is a custom dataset built over torch's DataLoader.
    The used datasets must subclass this class.

    Provides other modules an uniform way for accessing data provided by different
    datasets along some extra features.
    Implemented datasets must call `DataSource.__init__` so that `normalize`,
    `unnormalize` and `clamp` are correctly initialized.

    Args:
        mean (sequence): means for each channel as in `torchvision.transforms.Normalize`.
        str (sequence): standard deviations for each channel as in
            `torchvision.transforms.Normalize`.
        name (string): if `path` is not given, `name` is used to create a sub folder
            inside `data/datasets` and saves that path at `self.path`. Downloaded or
            manually added data should be placed in this folder.
        path (string): path to the dataset. Use this argument if you already have the
            dataset in your system and don't wish to relocate it.

    Note: either name or path must be provided.
    """
    def __init__(self, mean=None, std=None, name=None, path=None,  batch_size=8,
                 shuffle=None, num_workers=0, **kwargs):
        self.mean = mean
        self.std = std
        self.u_mean = None
        self.u_std = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = None
        self.min = None
        self.max = None

        if mean and std:
            # Defines new arrays u_mean and u_std to do the unnormalization
            self.u_mean = [(-mean[i] / std[i]) for i in range(len(mean))]
            self.u_std = [(1 / std[i]) for i in range(len(std))]

            # Sets min and max values of normalized images
            # Assumes the images before normalization are in the range [0,1]
            self.min = (0 - torch.tensor(mean)) / torch.tensor(std)
            self.max = (1 - torch.tensor(mean)) / torch.tensor(std)

        # Uses path if given or defines new one using the given name
        if path:
            self.path = path
        elif name:
            self.path = 'data/datasets/{}'.format(name)
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        else:
            raise ValueError("DataSource 'name' is required when 'path' is not given.")

    def get_dataset(self, train):
        """
        Returns the underlying dataset used.

        All custom datasets must implement this method. It must return the
        `torch.utils.data.Dataset` instance to use.

        Args:
            train (bool): argument that determines if it should get the training or
                testing dataset.

        Returns:
            torch.utils.data.Dataset: instance of the dataset to use.
        """
        raise NotImplementedError

    def get_classes(self):
        """
        Returns a list of human readable labels.

        Returns:
            list: list of strings with the labels of the dataset images.
        """
        raise NotImplementedError

    def get_dataloader(self, train=False):
        """
        Returns a `DataLoader` instance.

        Returns a testing or training `DataLoader` instance that uses the given
        `torch.utils.data.Dataset` instance returned by `get_dataset`.

        Args:
            train (bool): argument that determines if it should get the training or
                testing set.
        Returns:
            torch.utils.data.DataLoader: dataloader instance of the testing/training set.
        """
        dataset = self.get_dataset(train)
        dataloader = CustomDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if self.shuffle is not None else True,
            num_workers=self.num_workers,
            device=self.device)
        return dataloader

    def to(self, device):
        """
        'Sends' the dataset to device.

        Simulates having the dataset on device, images and labels returned by the
        `DataLoader` are automatically sent to device.

        Args:
            device (torch.device): device to which the returned images and labels
                will be automatically sent.
        """
        self.device = device

    def get_label(self, index):
        """
        Returns the human readable label for a given index.

        Args:
            index (int): index of the desired label.

        Returns:
            string: the human readable label for the passed index.
        """
        return self.get_classes()[index]

    def normalize(self, image, inplace=False):
        """
        Normalizes an image tensor with the passed mean and std.

        Args:
            image (Tensor): image tensor to be normalized
            inplace (Bool): Bool to make this operation inplace

        Returns:
            Tensor: the normalized tensor.
        """
        if self.mean and self.std:
            return normalize(image, self.mean, self.std, inplace)
        raise NotImplementedError("Normalization requires setting the mean and std.")

    def unnormalize(self, image, inplace=False):
        """
        Unnormalizes an image tensor with the passed mean and std.

        Args:
            image (Tensor): image tensor to be unnormalized
            inplace (Bool): Bool to make this operation inplace

        Returns:
            Tensor: the unnormalized tensor.

        """
        if self.u_mean and self.u_std:
            return normalize(image, self.u_mean, self.u_std, inplace)
        raise NotImplementedError("Unnormalization requires setting the mean and std.")

    def round_pixels_batch(self, images):
        rounded = torch.zeros_like(images)
        for i, img in enumerate(images):
            rounded[i] = self.unnormalize(img)
        rounded = (rounded * 255).round() / 255
        for i, img in enumerate(rounded):
            rounded[i] = self.normalize(img)
        return rounded

    def round_pixels(self, image):
        rounded = self.unnormalize(image)
        rounded = (rounded * 255).round() / 255
        return self.normalize(rounded)

    def clamp(self, tensor):
        """
        Clamps the elements in `tensor` to the dataset's normalized range.

        Uses the `min` and `max` of each channel (derived from the `mean` and `std`)
        to clamp the given tensor values.

        Args:
            tensor (Tensor): single or batch of image tensors.

        Returns:
            Tensor: The same tensor with it's elements clamped into the range [min, max].
        """
        if self.min is None or self.max is None:
            raise NotImplementedError("Clamp requires setting the mean and std.")

        dims = len(tensor.size())

        # Single min and max (images with one channel or same value for all channels)
        if len(self.min) == 1 and len(self.max) == 1:
            return torch.clamp(tensor, self.min.item(), self.max.item())

        # Single image
        if dims == 3 and len(self.min) == len(self.max) == tensor.size()[0]:
            channels = tensor.size()[0]
            t_min = self.min.view(channels, 1, 1).float().to(self.device)
            t_max = self.max.view(channels, 1, 1).float().to(self.device)
        # Batch of images
        elif dims == 4 and len(self.min) == len(self.max) == tensor.size()[1]:
            channels = tensor.size()[1]
            t_min = self.min.view(-1, channels, 1, 1).float().to(self.device)
            t_max = self.max.view(-1, channels, 1, 1).float().to(self.device)
        else:
            raise TypeError("Invalid tensor configuration")

        below = (tensor < t_min).float()
        over = (tensor > t_max).float()
        res = below * t_min + (1 - below) * tensor
        res = over * t_max + (1 - over) * res
        return res
