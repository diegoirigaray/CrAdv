from importlib import import_module


KNOWN_MODELS = {
    'resnet14_mnist': 'models.resnet14_mnist.ResNet',
    'resnet14_cifar': 'models.resnet14_cifar.ResNet_cifar10',
    'resnet50': 'models.resnet50.ResNet50',
    'vgg19': 'models.vgg19.VGG19',
    'densenet121': 'models.densenet121.DenseNet121',
    'inception_v3': 'models.inception_v3.InceptionV3',
}


KNOWN_DATASETS = {
    'mnist': 'datasources.mnist.MNIST',
    'fmnist': 'datasources.fmnist.FashionMNIST',
    'cifar10': 'datasources.cifar10.CIFAR10',
    'imagenet': 'datasources.imagenet.imagenet.ImageNet',
    'imagenet_inc3': 'datasources.imagenet.imagenet_inc3.ImageNetInception3',
}


KNOWN_ATTACKS = {
    'fgsm': 'attacks.fgsm.FGSM',
    'i_fgsm': 'attacks.i_fgsm.I_FGSM',
    'mi_fgsm': 'attacks.mi_fgsm.MI_FGSM',
    'deepfool': 'attacks.deepfool.DeepFool',
    'carlini_wagner': 'attacks.carliniwagner.CarliniWagner',
    'random_noise': 'attacks.random_noise.RandomNoise',
    'adversarial_sticker': 'attacks.adversarial_sticker.AdversarialSticker',
}


KNOWN_DEFENSES = {
    'jpeg': 'defenses.jpeg.JpegCompression',
    'crop_rescale': 'defenses.crop_rescale.CropRescale',
}


KNOWN_TASKS = {
    'accuracy': 'tasks.accuracy.Accuracy',
    'constrained_accuracy': 'tasks.constrained_accuracy.ConstrainedAccuracy',
    'samples': 'tasks.samples.Samples',
    'train': 'tasks.train.Train',
}


class Control(object):
    """
    Class used to access the different implementations inside `custom` folder.

    The different models, datasources, attacks, defenses and tasks implemented
    inside the `custom` folder must be added to the corresponding `KNOWN_****` dict.

    The selected key is one used in the tasks file to reference the class given by the
    string value, which must be the path in dot style to the desired class, relative to
    the `custom` folder.
    """
    def get_class(self, classes_dict, name):
        '''
        Imports and returns the specified class from the given classes dictionary.

        Args:
            classes_dict (object): dictionary of paths in dot notation.
            name (string): key of the desired element in `classes_dict`.

        Returns:
            class: class imported from the value of `name` element in `classes_dict`.
        '''
        path = classes_dict.get(name)
        if not path:
            raise Exception("'{}' was not found among {}.".format(name, classes_dict))

        path = path.split('.')
        class_name = path[-1]
        module = 'src.custom.' + '.'.join(path[:-1])
        return getattr(import_module(module), class_name)

    def get_model(self, name):
        """
        Returns the model specified by `name` in `KNOWN_MODELS` dict.
        """
        return self.get_class(KNOWN_MODELS, name)

    def get_datasource(self, name):
        """
        Returns the datasource specified by `name` in `KNOWN_DATASETS` dict.
        """
        return self.get_class(KNOWN_DATASETS, name)

    def get_attack(self, name):
        """
        Returns the attack specified by `name` in `KNOWN_ATTACKS` dict.
        """
        return self.get_class(KNOWN_ATTACKS, name)

    def get_defense(self, name):
        """
        Returns the defense specified by `name` in `KNOWN_DEFENSES` dict.
        """
        return self.get_class(KNOWN_DEFENSES, name)

    def get_task(self, name):
        """
        Returns the task specified by `name` in `KNOWN_TASKS` dict.
        """
        return self.get_class(KNOWN_TASKS, name)
