import torch
from src.base import Attack
from src.utils.functions import rescale_adversarial


class DeepFool(Attack):
    '''
    Implementation of the `Deepfool` attack.

    Reference:
        DeepFool: a simple and accurate method to fool deep neural networks
        https://arxiv.org/abs/1511.04599

    Extra args:
        eps (float): value used to reach the other side of the perturbation boundary.
        k_classes (int): number of classes considered in the perturbation search.
        max_iter (int): limit of iterations.
        max_2_norm (float): max euclidean norm for the perturbation. If given, the
            perturbation is constrained in each iteration to its value.
    '''
    def __init__(self, model, datasource, eps=0.02, k_classes=10, max_iter=50,
                 max_2_norm=None, dissimilarity=None, **kwargs):
        super().__init__(model, datasource, **kwargs)
        self.eps = eps
        self.k_classes = k_classes
        self.max_iter = max_iter
        self.max_2_norm = max_2_norm
        self.dissimilarity = dissimilarity

    def __call__(self, images, labels):
        pert_images = []

        if self.max_2_norm or self.dissimilarity:
            u_images = torch.stack([self.datasource.unnormalize(i) for i in images])
            u_norms = torch.norm(torch.flatten(u_images, start_dim=1), dim=1)

        # Process each image separately
        for i, image in enumerate(images):
            with torch.enable_grad():
                pert_image = None
                r_total = torch.zeros_like(image)

                # Does the forward with the starting image
                x_img = image[None,:].clone().detach().requires_grad_(True)
                f_image = self.model(x_img)

                # Obtains the top classes
                _, top_labels = torch.topk(f_image[0].clone(), self.k_classes)
                label = labels[i]
                true_in_top = (top_labels == label).nonzero()

                # If true label not in top k labels, return original image
                if len(true_in_top) != 1:
                    pert_images.append(image.clone())
                    continue

                k_true = true_in_top.item()

                iteration = 0
                k_i = label
                while k_i == label and iteration < self.max_iter:
                    pert = None
                    w = None

                    # Gets the gradient for the correct label
                    # Gets the gradient for the correct label\x_img
                    f_image[0, label].backward(retain_graph=True)
                    true_grad = x_img.grad.clone()

                    # Evaluates the distance to each of the top k classes
                    for k in range(self.k_classes):
                        if k == k_true:
                            continue
                        k_label = top_labels[k].item()
                        x_img.grad.zero_()

                        # Gets the gradient for the k label
                        f_image[0, k_label].backward(retain_graph=True)
                        k_grad = x_img.grad.clone()

                        w_k = k_grad - true_grad
                        f_k = f_image[0, k_label].data - f_image[0, label].data
                        pert_k = torch.abs(f_k) / torch.norm(w_k)

                        if pert is None or pert_k < pert:
                            pert = pert_k
                            w = w_k

                    # Obtains the current perturbation and adds it to the total perturbation
                    r_i = (pert + 1e-4) * w / torch.norm(w)
                    r_total = r_total + r_i[0]

                    # Rescales to constrain norm if given
                    if self.max_2_norm or self.dissimilarity:
                        adv = (image + r_total)[None,:]
                        u_img = u_images[i][None,:]
                        u_norm = u_norms[i][None]

                        if self.max_2_norm:
                            scaled = rescale_adversarial(
                                adv, u_img, u_norm, self.datasource, norm=self.max_2_norm, down_only=True)

                        elif self.dissimilarity:
                            scaled = rescale_adversarial(
                                adv, u_img, u_norm, self.datasource, self.dissimilarity, down_only=True)

                        pert_image = self.datasource.clamp(scaled[0])
                        r_total = pert_image - image
                    else:
                        pert_image = self.datasource.clamp(image + (1 + self.eps) * r_total)
                        r_total = pert_image - image

                    x_img = pert_image[None,:].clone().detach().requires_grad_(True)
                    f_image = self.model(x_img)
                    k_i = torch.argmax(f_image[0]).item()

                    iteration += 1
                pert_images.append(pert_image)

        return torch.stack(pert_images)
