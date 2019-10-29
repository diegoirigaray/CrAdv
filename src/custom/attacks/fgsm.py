import torch
import torch.nn as nn
from src.base import Attack


class FGSM(Attack):
    '''
    Implementation of the `Fast Gradient Sign Method` attack.

    Reference:
        Explaining and Harnessing Adversarial Examples
        https://arxiv.org/abs/1412.6572

    Extra Args:
        eps (float): value used to rescale the sign tensor. If not given, searches the
            minimum epsilon that manages to fool the network.
        try_eps_min (float): minimum epsilon tested when `eps` is not given.
        try_eps_max (float): maximum epsilon tested when `eps` is not given.
        try_eps_step (float): step used to increment the tested epsilon between
            `try_eps_min` and `try_eps_max` when `eps` is not given.
    '''
    def __init__(self, model, datasource, eps=None, try_eps_min=0.001, try_eps_max=1,
                 try_eps_step=0.001, **kwargs):
        super().__init__(model, datasource, **kwargs)
        self.eps = eps
        self.try_eps_min = try_eps_min
        self.try_eps_max = try_eps_max
        self.try_eps_step = try_eps_step

    def __call__(self, images, labels):
        batch_size = images.size()[0]
        criterion = nn.CrossEntropyLoss()

        with torch.enable_grad():
            x = images.clone().detach().requires_grad_(True)
            outputs = self.model(x)
            targets = torch.argmax(outputs.data, 1)

            loss = criterion(outputs, targets)
            loss.backward()
            x_grad = torch.sign(x.grad.data)

            # If an epsilon is given, use it
            if self.eps is not None:
                adv_x = x.data + self.eps * x_grad
                adv_x = self.datasource.clamp(adv_x)
            # If not, evaluates epsilons between `try_eps_min` and `try_eps_max`
            # and use the forst one that fools the net (if any)
            else:
                adv_x = torch.zeros_like(images)
                working_indexes = torch.tensor(range(batch_size)).to(
                    self.datasource.device)
                current_eps = self.try_eps_min

                while current_eps <= self.try_eps_max and len(working_indexes):
                    # Obtains the perturbations
                    current_adv = (torch.index_select(images, 0, working_indexes) +
                                   torch.index_select(x_grad, 0, working_indexes) *
                                   current_eps)
                    current_adv = self.datasource.clamp(current_adv)
                    outputs = self.model(current_adv)
                    _, predicted = torch.max(outputs.data, 1)

                    # Gets the images that fooled the net and adds each one to
                    # the batch of adversarial samples `adv_x`
                    fooled = (predicted !=
                              torch.index_select(labels, 0, working_indexes))
                    for i in range(len(fooled)):
                        if fooled[i].item() == 1:
                            adv_x[working_indexes[i]] = current_adv[i]

                    # Updates the `working_indexes` to contain only the indexes of
                    # images that havent fooled the net yet
                    working_indexes = torch.index_select(
                        working_indexes, 0, torch.flatten((fooled == 0).nonzero()))

                    current_eps += self.try_eps_step

                if len(working_indexes):
                    current_adv = (torch.index_select(images, 0, working_indexes) +
                                   torch.index_select(x_grad, 0, working_indexes) *
                                   current_eps)
                    for i in range(len(working_indexes)):
                        adv_x[working_indexes[i]] = current_adv[i]

            return adv_x
