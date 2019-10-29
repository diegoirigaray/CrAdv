import torch
import torch.nn as nn
from src.base import Attack
from src.utils.functions import rescale_adversarial


class I_FGSM(Attack):
    '''
    Iterative version of the gradient sign method.

    Extra Args:
        eps (float): value used to limit the perturbation size (max inf norm).
        alpha (float): value used to rescale the sign vector at each iteration.
            If not given uses alpha = 1.25 * eps / iterations
        iterations (int): number of iterations.
        max_2_norm (float): max euclidean norm for the perturbation. If given, the
            perturbation is constrained in each iteration to its value.
    '''
    def __init__(self, model, datasource, eps=0.1, alpha=None, iterations=10, max_2_norm=None,
                 dissimilarity=None, least_likely=False, min_alpha=None, **kwargs):
        super().__init__(model, datasource, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.iterations = iterations
        self.max_2_norm = max_2_norm
        self.dissimilarity = dissimilarity
        self.least_likely = least_likely
        self.min_alpha = min_alpha

    def project(self, perts):
        # Else, projects the perturbation to a ball of radius `eps`
        over = (perts > self.eps).float()
        res = over * self.eps + (1 - over) * perts
        below = (res < -self.eps).float()
        res = below * -self.eps + (1 - below) * res
        return res

    def rescale(self, adversarial):
        if self.max_2_norm:
            return rescale_adversarial(adversarial, self.u_images, self.u_norms,
                                       self.datasource, norm=self.max_2_norm, down_only=True)
        return rescale_adversarial(adversarial, self.u_images, self.u_norms, self.datasource,
                                   self.dissimilarity, down_only=True)

    def __call__(self, images, labels):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        exec_alpha = self.alpha if self.alpha else 1.25 * self.eps / self.iterations
        if self.min_alpha and exec_alpha < self.min_alpha:
            exec_alpha = self.min_alpha

        if self.max_2_norm or self.dissimilarity:
            self.u_images = torch.stack([self.datasource.unnormalize(i) for i in images])
            self.u_norms = torch.norm(torch.flatten(self.u_images, start_dim=1), dim=1)

        if not self.least_likely:
            grad_factor = 1
            loss_labels = labels
        else:
            grad_factor = -1
            outputs = self.model(images)
            _, indexes = torch.topk(outputs, 1, dim=1, largest=False)
            loss_labels = torch.flatten(indexes)

        with torch.enable_grad():
            adv_x = images.clone().detach().requires_grad_(True)
            perts = torch.zeros_like(images)
            outputs = self.model(adv_x)

            for i in range(self.iterations):
                loss = criterion(outputs, loss_labels)
                loss.backward()
                x_grad = torch.sign(adv_x.grad.data)
                adv_x.grad.zero_()

                perts = perts + grad_factor * exec_alpha * x_grad

                if self.max_2_norm or self.dissimilarity:
                    adversarial = self.datasource.clamp(self.rescale(perts + images))
                else:
                    perts = self.project(perts)
                    adversarial = images + perts

                adversarial = self.datasource.clamp(adversarial)
                perts = adversarial - images
                adv_x = adversarial.clone().detach().requires_grad_(True)
                outputs = self.model(adv_x)
        return adv_x
