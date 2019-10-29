import math
from torchvision.models.densenet import DenseNet, _load_state_dict, model_urls


class DenseNet121(DenseNet):

    def load_weights(self):
        _load_state_dict(self, model_urls['densenet121'], True)
