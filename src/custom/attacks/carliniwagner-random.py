import math
import torch
from src.base import Attack
from src.utils.functions import to_neg_one_one, from_neg_one_one
import random

class CarliniWagner(Attack):
    """
    Implementation of the L2 version of the Carlini & Wagner attack

    Reference:
        Towards Evaluating the Robustness of Neural Networks
        https://arxiv.org/abs/1608.04644

    Extra args:
        bin_search_steps (int): number of iterations of the binary search for the c const.
        max_iterations (int): number of iterations of the perturbation search for
            each c const.
        abort_early (bool): flag to enable early abort when not enough progress.
        c_init (float): initial value for the c const.
        learning_rate (float): learning rate using on the optimizer.
        confidence (float): constant used to regulate the confidence of the attack.
    """

    def __init__(self, model, datasource, bin_search_steps=10, max_iterations=1000,
                 abort_early=True, c_init=1e-2, c_const=None, learning_rate=1e-2,
                 confidence=0, target_class=None, random_classes=None, **kwargs):

        super().__init__(model, datasource, **kwargs)
        self.bin_search_steps = bin_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.num_classes = len(datasource.get_classes())
        self.device = datasource.device
        self.c_init = c_init
        self.c_const = c_const
        self.learning_rate = learning_rate
        self.confidence = confidence
        self.target_class = target_class
        self.random_classes = random_classes

        # Boolean that indicates if the images are already in the range [-1, 1]
        self.images_ready = (len(self.datasource.min) == len(self.datasource.max) == 1 and
                             self.datasource.min.item() == -1 and
                             self.datasource.max.item() == 1)

    def to_w(self, x):
        if not self.images_ready:
            x = to_neg_one_one(x, self.datasource.min, self.datasource.max)

        # Implementation of arctanh
        res = x * 0.999999
        return 0.5 * torch.log((1 + res) / (1 - res))

    def from_w(self, x):
        res = x.tanh()

        if not self.images_ready:
            res = from_neg_one_one(res, self.datasource.min, self.datasource.max)
        return res

    def loss_func(self, x_orig, x_pert, const, orig_label, target_labels, logits):
        flat = torch.flatten((x_pert - x_orig), start_dim=1)
        distance_loss = torch.norm(flat, dim=1) ** 2

        adv_loss = torch.zeros_like(distance_loss)
        for i, target in enumerate(target_labels):
            correct = orig_label[i]
            adv_loss[i] = max(0, self.confidence + logits[i][correct] - logits[i][target])

        total_loss = distance_loss + const * adv_loss
        return torch.sum(total_loss)

    def __call__(self, images, labels):
        logits = self.model(images)
        target_labels = []
        
        if (self.target_class is None):
            # Use the second most likely classes that aren't correct
            _, indexes = torch.topk(logits, 2, dim=1)
            for iter, ind in enumerate(indexes):
                correct = labels[iter].item()
                first = ind[0].item()
                second = ind[1].item()
                target_labels.append(first if first != correct else second)
        elif (self.target_class == "random"): 
            classes = list(range(self.num_classes))
            if self.random_classes:
                classes = [int(n) for n in self.random_classes.split()]
            for iter, ind in enumerate(logits):
                correct = labels[iter].item()
                random_class = random.choice(classes)
                while random_class == correct:
                    random_class = random.choice(classes)
                target_labels.append(random_class)
        elif (self.target_class == "least_likely"): 
            # If least_likely is True, targets the classes with less probability
            _, indexes = torch.topk(logits, 1, dim=1, largest=False)
            target_labels = [i.item() for i in indexes]
        else:
            raise NotImplementedError("Targetted method not implemented.")
            
        if self.c_const:
            bin_steps = 1
            c_value = self.c_const
        else:
            bin_steps = self.bin_search_steps
            c_value = self.c_init

        with torch.enable_grad():
            shape = labels.size()
            result = images.clone()
            c_const = torch.empty(shape).fill_(c_value).to(images.device)
            c_min = torch.empty(shape).fill_(float("Inf")).to(images.device)
            lower_bound = torch.empty(shape).fill_(0.).to(images.device)
            upper_bound = torch.empty(shape).fill_(float("Inf")).to(images.device)

            # Binary steps to find best c const
            for step in range(bin_steps):
                found_adv = torch.empty(shape).fill_(0.0).to(images.device)
                prev_loss = None

                # In the last step use the upper bound
                if step == self.bin_search_steps - 1:
                    for i in range(len(c_const)):
                        is_inf = upper_bound[i] == float("Inf")
                        c_const[i] = c_const[i] if is_inf else upper_bound[i]

                # Creates initial perturbation and optimizer
                w_pert = self.to_w(images).clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([w_pert], lr=self.learning_rate)
                for s_iter in range(self.max_iterations):
                    optimizer.zero_grad()
                    x_pert = self.from_w(w_pert)

                    logits = self.model(x_pert)
                    loss = self.loss_func(images, x_pert, c_const, labels, target_labels,
                                          logits)
                    loss.backward()
                    optimizer.step()

                    found_adv = (torch.argmax(logits, 1) != labels).float()

                    dec = s_iter % (math.ceil(self.max_iterations / 10)) == 0
                    if self.abort_early and dec:
                        if prev_loss and not (loss.item() <= .9999 * prev_loss):
                            print("break in " + str(step) + " in iteration " + str(s_iter))
                            break
                        prev_loss = loss.item()

                for i in range(len(c_const)):
                    # Update upper bound
                    if found_adv[i].item() == 1:
                        upper_bound[i] = c_const[i]

                    # Update lower bound
                    if found_adv[i].item() == 0:
                        lower_bound[i] = c_const[i]

                    # Update adversarial samples
                    if found_adv[i] and c_const[i].item() < c_min[i].item():
                        result[i] = x_pert[i]
                        c_min[i] = c_const[i]

                    # Update const
                    if upper_bound[i].item() == float("Inf"):
                        c_const[i] *=  10
                    else:
                        c_const[i] = (lower_bound[i] + upper_bound[i]) / 2
        return result
