import math
from torchvision.models.vgg import VGG, make_layers, model_urls, cfgs


class VGG19(VGG):
    model_url = model_urls['vgg19_bn']

    def __init__(self, **kwargs):
        super().__init__(make_layers(cfgs['E'], True))
