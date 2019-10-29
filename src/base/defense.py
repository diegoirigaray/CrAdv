import os
import torch.nn as nn
from torch.autograd import Function


class DummyFunction(Function):
    '''
    Dummy differentiable function to use as example
    '''
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Defense(nn.Module):
    '''
    Base class for all defenses.

    Your defenses must subclass this class.

    Defenses must behave as if they were models themselves, receiving images and returning
    the estimated labels. We contemplate two types of defenses: model-agnostic defenses
    that apply some pre-processing to the images (like for example JPEG compression) and
    model-specific defenses (like defensive distillation).

    For the model-agnostic case, defenses only need to provide the `process` method.
    The model-specific defenses must re-implement the `forward` method.

    Args:
        model (nn.Module): model to be defended.
        datasource (base.datasource.DataSource): datasource, normally the one on
            which `model` is trained.
    '''

    def __init__(self, model, datasource, **kwargs):
        super().__init__()
        self.model = model
        self.datasource = datasource

    def forward(self, input):
        '''
        Returns the images classification with the defense applied.

        The base implementation assumes a model-agnostic defense, calls the `process`
        function and uses the resulting images on the given `model`.
        For model-specific defenses re-implement this method.

        Args:
            input (Tensor): images (potentially perturbed) to classify.

        Returns:
            Tensor: estimated labels.
        '''
        # Clones the input to avoid modifying it outside of this function
        input = input.clone()

        out = self.process(input)
        out = self.model(out)
        return out

    def process(self, input):
        '''
        Process the images to apply model-agnostic defenses.

        Model-agnostic defenses must re-implement this method.

        In order to make your defense differentiable consider using
        `torch.autograd.functions.Function` to hold your defense logic.

        Args:
            input (Tensor): images (potentially perturbed) to classify.

        Returns:
            Tensor: the same input images with the defense applied to them.
        '''
        return DummyFunction.apply(input)
