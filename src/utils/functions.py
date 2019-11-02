import os
import torch
from time import time


WEIGHTS_PATH = "data/weights"


class Timer():
    def start(self):
        self.start_time = time()

    def stop(self):
        end_time = time()
        return time_to_string(self.start_time, end_time)


def to_percentage(f):
    return "{:.2f}%".format(100 * f)


def save_weights(net, net_id):
    '''
    Saves the net weights given a net_id

    Args:
        net (nn.Module): model for which to save the weights.
        net_id (string): identifier of the model weights. Used as name for the weights
            file, and should be passed in later executions to allow loading it.
    '''
    # Creates the folder to store the weights if it does not exist
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)

    # Creates weights file
    w_path = "{}/{}.pth".format(WEIGHTS_PATH, net_id)
    open(w_path, 'w')

    # Stores trained net weights
    torch.save(net.state_dict(), w_path)


def load_weights(net, net_id):
    '''
    Loads the net weigths identified by the passed net_id

    Args:
        net (nn.Module): model for which to the weights are loaded.
        net_id (string): identifier of the model weights. Same net_id used when
            saving the weights.
    '''
    w_path = "{}/{}.pth".format(WEIGHTS_PATH, net_id)
    try:
        weights = torch.load(w_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(weights)
        return True
    except FileNotFoundError:
        print("**Weights for net_id: {} were not found**".format(net_id))
        return False


def time_to_string(start, end):
    '''
    Gets the duration between two times.

    Given a starting and an ending time, calculates the duration between the two and
    outputs it in a human readable way.

    Args:
        start (float): Float representing the starting time
        end (float): Float representing the ending time

    Returns:
        string: duration between start and end specified as hours, minutes and seconds.
    '''
    total = int(end - start)
    return "{}h {}m {}s".format(total // 3600, (total % 3600) // 60, total % 60)


def rescale_to_norm(tensor, norm, down_only=False):
    '''
    Rescales the given images to have the given norm.

    Rescales either a single image tensor, or a batch of them so that each image ends up
    with an euclidean norm equal to the given `norm`.

    Args:
        tensor (Tensor): single or batch of image tensors to rescale.
        norm (Number): desired euclidean norm.
        down_only (Bool): flag, if true images will be rescaled only if their initial
            norm is greater than the desired one, if false, images will also be scaled
            up to match the given norm.

    Returns:
        Tensor: tensor of the same dimensions of the input one, with each image having
            an euclidean norm of `norm`.
    '''
    dims = len(tensor.size())

    # Single image
    if dims == 3:
        t_norm = torch.norm(tensor)

        # If tensor norm is already smaller and `down_only` is True return tensor
        if t_norm == 0 or (t_norm < norm and down_only):
            return tensor

        tensor = tensor * (norm / t_norm)
        return tensor

    if dims == 4:
        t_norm = torch.norm(torch.flatten(tensor, start_dim=1), dim=1)
        factor = norm / t_norm

        # If `down_only` tops factors larger than 1 to 1
        if down_only:
            factor = torch.clamp(factor, 0, 1)
        tensor = factor.view(-1, 1, 1, 1) * tensor
        return tensor

    raise TypeError("tensor dimensions not valid")


def rescale_adversarial(adversarial, u_clean, u_clean_norm, datasource, dissimilarity=None,
                        norm=None, down_only=False):
    '''
    Rescales perturbations in a batch of images.

    Rescales the perturbations of a batch of images to either an specified norm
    or normalized dissimilarity.
    The norm or normalized dissimilarity is obtained in the unnormalized range [0,1]
    but the adversarial batch is expected and returned in the normmalized one.

    Args:
        adversarial (Tensor): batch of adversarial normalized images.
        u_clean (Tensor): batch of the corresponding clean unnormalized images.
        u_clean_norm (Tensor): norms of the `u_clean` images, expected as argument
            for improve performance.
        datasource (DataSource): corresponding DataSource instance, used to
            normalize and unnormalize images.
        dissimilarity (numeric, optional): if given, the perturbations will be
            rescaled to get this value as normalized dissimilarity.
        norm (numeric, optional): if `dissimilarity` is not given, the perturbations
            are rescaled to this `norm` value (euclidean).
        down_only (bool): if `True`, the perturbations will only be down scaled.

    Returns:
        Tensor: adversarial images fromm the `adversarial` input but with their
            perturbations scaled to match the given `dissimilarity` or `norm`.
    '''
    results = []
    if dissimilarity:
        scaled_norms = u_clean_norm * dissimilarity

    for i, adv in enumerate(adversarial):
        u_adv = datasource.unnormalize(adv)
        pert = u_adv - u_clean[i]
        pert = rescale_to_norm(pert, scaled_norms[i] if dissimilarity else norm, down_only)
        results.append(datasource.normalize(u_clean[i] + pert))

    return torch.stack(results)


def get_a_b(t_size, min_, max_):
    '''
    Auxiliar function used by `to_neg_one_one` and `from_neg_one_one` methods.
    '''
    # If min and max are single values
    if len(min_) == len(max_) == 1:
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
    # If there are different min and max values for each channel
    else:
        # For single images
        if len(t_size) == 3 and len(min_) == len(max_) == t_size[0]:
            a = ((min_ + max_) / 2).view(t_size[0], 1, 1)
            b = ((max_ - min_) / 2).view(t_size[0], 1, 1)
        # For batch of images
        elif len(t_size) == 4 and len(min_) == len(max_) == t_size[1]:
            a = ((min_ + max_) / 2).view(-1, t_size[1], 1, 1)
            b = ((max_ - min_) / 2).view(-1, t_size[1], 1, 1)
        else:
            raise TypeError("Invalid tensor dimension")
    return a.float(), b.float()


def to_neg_one_one(tensor, min_, max_):
    '''
    Convert a single or batch of image tensors in the range [min_, max_] to [-1, 1]

    Args:
        tensor (Tensor): single image tensor or batch of images to convert.
        min_ (Tensor): tensor with the minimum values for each channel of the
            given tensor image/s.
        max_ (Tensor): tensor with the maximum values for each channel of the
            given tensor image/s.

    Returns:
        Tensor of the same dimensions of the input tensor, with its elements in
        the range [-1, 1]
    '''
    a, b = get_a_b(tensor.size(), min_, max_)
    return (tensor - a.to(tensor.device)) / b.to(tensor.device)


def from_neg_one_one(tensor, min_, max_):
    '''
    Convert a single or batch of image tensors in the range [-1, 1] to [min_, max_]

    Args:
        tensor (Tensor): single image tensor or batch of images to convert.
        min_ (Tensor): tensor with the minimum values for each channel of the
            desired tensor image/s.
        max_ (Tensor): tensor with the maximum values for each channel of the
            desired tensor image/s.

    Returns:
        Tensor of the same dimensions of the input tensor, with its elements in
        the range [min_, max_]
    '''
    a, b = get_a_b(tensor.size(), min_, max_)
    return tensor * b.to(tensor.device) + a.to(tensor.device)
