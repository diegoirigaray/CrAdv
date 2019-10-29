import os


class Attack(object):
    """
    Base class for all attacks.

    Your attacks must subclass this class.

    If you would like to execute tasks varying some of the attack's hyperparameters,
    use it as an object attribute, this allows `Scheduler` to change its values
    between executions using the `set_attr` method in the attack (see `attack_variables`
    in the task schema).
    If changing certain parameter needs to perform any extra processing, it must be
    implemented on a specific method: `set_[parameter](self, value)` (for example
    `set_perturbation_norm`).

    Args:
        model (torch.nn.Module or base.defense.Defense): model used to obtain the
            adversarial images. Either the currently used defense instance, it's
            underlying model or a substitute model (see `attack_on_defense`,
            `on_task_model` and `specific_models` at `utils.schemas`).
        datasource (base.datasource.DataSource): same datasource passed to the `Task`.
    """

    def __init__(self, model, datasource, **kwargs):
        self.model = model
        self.datasource = datasource

    def __call__(self, images, labels):
        """
        Applies the attack to a set of images.

        All attacks must implement this method, it performs the attack it self.

        Args:
            images (Tensor): tensor of images to attack.
            labels (Tensor): the `images` corresponding labels.

        Returns:
            Tensor: perturbed images
        """
        raise NotImplementedError

    def set_attr(self, name, value):
        '''
        Sets the given value as an object attribute.

        Used by `Scheduler` to alter an attack's behaviour dinamically.
        By default uses `setattr`, to add extra logic implement a method
        called `set_[name]` that receives the desired value.

        Args:
            name (string): name of the attribute
            value: new value
        '''
        if hasattr(self, "set_{}".format(name)):
            getattr(self, "set_{}".format(name))(value)
        else:
            setattr(self, name, value)
