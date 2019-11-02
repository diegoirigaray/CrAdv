import torch
import torch.nn as nn
from src.base import Attack


class MI_FGSM(Attack):
    '''
    Implementation of the Momentum Iterative FGSM.

    Extra Args:
        eps (float): value used to limit the perturbation size (max inf norm).
        iterations (int): number of steps in the gradient descent.
        decay_factor (float): decay factor used with the velocity vector.
        max_2_norm (float): max euclidean norm for the perturbation. If given, the
            perturbation is constrained in each iteration to its value.
    '''
    def __init__(self, model, datasource, eps=0.1, iterations=10, decay_factor=1.0,
                 max_2_norm=None, **kwargs):
        super().__init__(model, datasource, **kwargs)
        self.eps = eps
        self.iterations = iterations
        self.decay_factor = decay_factor
        self.max_2_norm = max_2_norm

    def __call__(self, images, labels):
        self.model.eval()
        alpha = self.eps / self.iterations
        criterion = nn.CrossEntropyLoss()

        with torch.enable_grad():
            adv_x = images.clone().detach().requires_grad_(True)
            pert = torch.zeros_like(images)
            g_vector = torch.zeros_like(images)
            outputs = self.model(adv_x)

            for i in range(self.iterations):
                loss = criterion(outputs, labels)
                loss.backward()
                x_grad = torch.sign(adv_x.grad.data)
                adv_x.grad.zero_()

                # Update the velocity vector
                flat = torch.flatten(x_grad, start_dim=1)
                normalized = flat / torch.reshape(torch.norm(flat, 1, 1), [len(flat), 1])
                g_vector = g_vector * self.decay_factor + torch.reshape(normalized,
                                                                        x_grad.size())

                # Update the perturbation vector
                pert = pert + alpha * torch.sign(g_vector)
                adv_x = self.datasource.clamp(
                    images + pert).clone().detach().requires_grad_(True)
                outputs = self.model(adv_x)
        return adv_x
