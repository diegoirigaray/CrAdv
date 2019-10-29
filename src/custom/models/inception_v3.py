from torchvision.models.inception import Inception3, model_urls


class InceptionV3(Inception3):
    model_url = model_urls['inception_v3_google']
