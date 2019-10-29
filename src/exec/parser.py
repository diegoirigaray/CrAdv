import torch
import torch.nn as nn
from datetime import datetime
from src.control import Control
from torchvision.models.utils import load_state_dict_from_url
from src.utils.functions import save_weights, load_weights


class Parser(object):
    """
    Converts the info of a tasks file into objects expected by `Scheduler`.

    Parses the info specified in the tasks file for the following components
    into objects expected by the `Scheduler` class:
    - config
    - net
    - defense
    - attack
    - task

    Sets default values, and except for the config element, instantiates the
    corresponding classes and adds a `data` attribute containing its basic info.
    `data` is mainly used for saving the tasks results.

    Note:   in each case, the `****_params` attribute is passed as kwargs when
            instantiating the class.

    For a description of the posible arguments see `schemas.py` in utils folder.
    """
    def __init__(self):
        self.control = Control()

    def parse_config(self, config):
        """
        Sets the default configuration values for the execution of a tasks file.

        By default uses all available devices and stores the results on the
        `results` folder, on a sub folder named with the current date time.

        Args:
            config (object): object specified by `config` in `schemas.schedule_schema`.
        """
        # Set device/s to use
        if torch.cuda.is_available():
            default_device = torch.device("cuda:0")
        else:
            default_device = torch.device("cpu")
        config.setdefault("device", default_device)
        config.setdefault("device_ids", None)
        config.setdefault("batch_size_factor", 1)
        config.setdefault("safe_mode", False)

        device_count = torch.cuda.device_count()
        config['multi_gpu'] = config.get('multi_gpu', True) and device_count > 1

        # Set results path
        default_results_path = "results/{}".format(
            datetime.strftime(datetime.now(), '%y-%m-%d_%R'))
        config.setdefault("results_path", default_results_path)

        return config


    def parse_task(self, task_data):
        """
        Crates a `Task` object with the specified info.

        By default the task is executed for all posible combinations of *model*,
        *defended model*, *no attack*, *attack* and *attack with variables*.
        To alter this behaviour set the corresponding `with_****` or `without_****`
        flags to `false`.

        Args:
            task_data (object): object specified by `task_data` in `schemas.task_schema`.

        Returns:
            base.task.Task
        """
        # Set task default values
        task_data.setdefault("task_params", {})
        task_data.setdefault("exec_multi", False)
        task_data.setdefault("attack_on_defense", True)
        task_data.setdefault("plot_results", False)
        task_data.setdefault("plot_keys", [])
        task_data.setdefault("plot_together", True)
        task_data.setdefault("skip_no_defense", False)
        task_data.setdefault("skip_no_attack", False)
        task_data.setdefault("skip_no_attack_variables", False)

        # Instantiates task
        task_cls = self.control.get_task(task_data['task_name'])
        task = task_cls(**task_data['task_params'])
        task.data = task_data
        return task


    def parse_net(self, net_data, config):
        """
        Instantiates an `nn.Module` model, a `DataSource` and loads the model weights.

        Creates the specified model and load its weights if they exist. The weights
        must be stored in the `data/weights` folder, in a file named as the
        `net_id` attribute and pth extension (as if saved by the
        `utils.functions.save_weights` function).

        Also instantiates the `DataSource` object (usually the one used to train
        the model)

        Args:
            net_data (object): object specified by the `schemas.net_schema`.

        Returns:
            nn.Module
            base.datasource.DataSource
        """
        net_data.setdefault("model_params", {})
        net_data.setdefault("datasource_params", {})

        net_data['datasource_params'].setdefault("batch_size", 8)
        net_data['datasource_params']['batch_size'] = int(
            net_data['datasource_params']['batch_size'] *
            config["batch_size_factor"])

        # Instantiates the net and loads its weights
        model_cls = self.control.get_model(net_data['model_name'])
        net = model_cls(**net_data['model_params'])

        # Tries to load the weights for the given net_id
        # If weights not found, and the model has a 'model_url' attribute,
        # uses it to download the weights and store them
        if not load_weights(net, net_data['net_id']):
            if hasattr(net, 'model_url'):
                state_dict = load_state_dict_from_url(net.model_url)
                net.load_state_dict(state_dict)
                save_weights(net, net_data['net_id'])
            elif hasattr(net, 'load_weights'):
                net.load_weights()
                save_weights(net, net_data['net_id'])

        # Instantiates the datasource
        datasource_cls = self.control.get_datasource(net_data['datasource_name'])
        datasource = datasource_cls(
            **net_data['datasource_params'],
            batch_size_factor=config["batch_size_factor"])

        # Sends the model and the dataset to device
        net.to(config["device"])
        datasource.to(config["device"])
        net.eval()

        # If `multi_gpu` is enabled, adds DataParallel
        if config["multi_gpu"]:
            net = nn.DataParallel(net, config['device_ids'])

        net.data = net_data
        return net, datasource


    def parse_defense(self, net, datasource, defense_data):
        """
        Creates an instance of `Defense` for a fiven model and datasource.

        Args:
            net (nn.Module): model to 'defend'
            datasource (base.datasource.DataSource): datasource, normally the one used to
                train the model.
            defense_data (object): items from `defenses` in `schemas.task_schema`

        Returns:
            base.defense.Defense
        """
        defense_data.setdefault("defense_params", {})

        defense_cls =  self.control.get_defense(defense_data['defense_name'])
        defense = defense_cls(net, datasource, **defense_data['defense_params'])
        defense.data = defense_data
        return defense


    def parse_attack(self, net, datasource, attack_data):
        """
        Creates an instance of `Attack` for a fiven model and datasource.

        Args:
            net (torch.nn.Module or base.defense.Defense): either the currently
                used defense instance, it's underlying model or a substitute model.
            datasource (base.datasource.DataSource): the datased specified with the net for the
                current task execution, in case its needed.
            attack_data (object): items from `attacks` in `schemas.task_schema`

        Returns:
            base.attack.Attack
        """
        attack_data.setdefault("attack_params", {})
        attack_data.setdefault("except_variables", [])
        attack_data.setdefault("on_task_model", True)
        attack_data.setdefault("specific_models", [])

        attack_cls = self.control.get_attack(attack_data['attack_name'])
        attack = attack_cls(net, datasource, **attack_data['attack_params'])
        attack.data = attack_data
        return attack


class DefenseIterable(object):
    """
    Iterable of the specified defenses.

    Converts the defenses specification of the task's file into an iterator of
    `src.base.defense.Defense` instances, using the
    `src.exec.parser.Parser.parse_defense` method.

    Args:
        skip_no_defense (bool): flag that when False, causes the first returned element
            to be `None`, thus executing the task without defense.
        net (nn.Module): model to use on the defenses.
        datasource (base.datasource.DataSource): datasource to use on the defenses.
        defenses (object): `defenses` array from `schemas.task_schema`

    Returns:
        Iterator of `src.base.defense.Defense` instances.
    """
    def __init__(self, skip_no_defense, net, datasource, defenses):
        self.parser = Parser()
        self.skip_no_defense = skip_no_defense
        self.return_none = not skip_no_defense;
        self.net = net
        self.datasource = datasource
        self.defenses = defenses
        self.current = 0

    def count(self):
        count = len(self.defenses)
        if not self.skip_no_defense:
            count += 1
        return count

    def _reset(self):
        self.current = 0
        self.return_none = not self.skip_no_defense;

    def __iter__(self):
        return self

    def __next__(self):
        # If execution without defense is enabled, first return None
        if self.return_none:
            self.return_none = False
            print("**No Defense**")
            return None
        # Returns the corresponding defense instance
        if self.current < len(self.defenses):
            defense = self.parser.parse_defense(self.net, self.datasource,
                                                self.defenses[self.current])
            self.current += 1
            print("**Defense: {}**".format(defense.data["defense_name"]))
            return defense
        self._reset()
        raise StopIteration


class AttackIterable(object):
    """
    Iterable of the specified attacks.

    Converts the attacks specification of the task's file into an iterator of
    `src.base.attack.Attack` instances, using the
    `src.exec.parser.Parser.parse_attack` method.

    Args:
        skip_no_attack (bool): flag that when true, causes the first returned element
            to be `None`, thus executing the task without attack.
        net (nn.Module): model used to create the adversary samples (may also be a
            base.defense.Defense instance).
        datasource (base.datasource.DataSource): datasource of the current task execution.
        attacks (object): `attacks` array from `schemas.task_schema`

    Returns:
        Iterator of `src.base.attack.Attack` instances.
    """
    def __init__(self, skip_no_attack, net, datasource, attacks, config):
        self.parser = Parser()
        self.skip_no_attack = skip_no_attack;
        self.return_none = not skip_no_attack;
        self.net = net
        self.datasource = datasource
        self.attacks = attacks
        self.current = 0
        self.sub_current = -1
        self.config = config

    def count(self):
        count = len(self.attacks)
        if not self.skip_no_attack:
            count += 1
        return count

    def _reset(self):
        self.current = 0
        self.sub_current = -1
        self.return_none = not self.skip_no_attack;

    def __iter__(self):
        return self

    def set_defense(self, defense):
        # Sets the current defense if the `AttackIterable` was initialized with a model.
        self.net = defense

    def __next__(self):
        # If execution without attack is enabled, first return None
        if self.return_none:
            self.return_none = False
            print("**No Attack**")
            return None
        # Returns the corresponding attack instance
        if self.current < len(self.attacks):
            attack_data = self.attacks[self.current]
            specific_models = attack_data.get("specific_models", [])

            # If attacking with the task model is enabled, instantiates it
            if attack_data.get("on_task_model", True) and self.sub_current == -1:
                attack = self.parser.parse_attack(self.net, self.datasource,
                                                  attack_data)
            # Instantiates the attack with the corresponding model
            else:
                s_net, s_datasource = self.parser.parse_net(
                    specific_models[self.sub_current], self.config)
                attack = self.parser.parse_attack(s_net, s_datasource, attack_data)

            # Updates the indexes
            if self.sub_current + 1 < len(specific_models):
                self.sub_current += 1
            else:
                self.sub_current = -1
                self.current += 1

            name = "model_name" if "model_name" in attack.model.data else "defense_name"
            print("Attack: {} on model: {}".format(attack.data["attack_name"],
                                                   attack.model.data[name]))
            return attack

        self._reset()
        raise StopIteration


def filter_attack_variables(attack, a_var):
    if not attack:
        return []
    except_var = attack.data['except_variables']
    res = [
        (a['variable_name'], a['variable_values'])
        for a in a_var if a['variable_name'] not in except_var]
    return res
