from torchvision.models.resnet import ResNet, Bottleneck, model_urls


class ResNet50(ResNet):
    model_url = model_urls['resnet50']

    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])
