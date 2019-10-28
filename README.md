
# CrAdv

CrAdv is a Python framework built around PyTorch and designed to make it easier for researchers to implement and test adversarial attacks and defenses.
It allows users to focus on the implementation of their algorithms and quickly test them and compare them with existing ones. It also provides an environment on where to easily share code and make it more accessible to reproduce experiments.

## Getting Started

To get started simply clone the repository to your machine and install the ``requirements.txt`` file, preferably in its own virtualenv.
```
# Create and activate the virtualenv
virtualenv -p python3 cradv
source cradv/bin/activate

# Install dependecies
pip install -r requirements.txt
```

In the following sections we describe how to use this framework. First we explain how to run experiments (also referred as *tasks file*) using the implemented attacks, defenses, etc. Then we give an overview of the user defined components and how to implement new ones.


## Execution

All code we would execute using the implemented components is encapsulated in the so called ``Tasks``, which are basically pieces of code that given a model, dataset, attack and defense give back some kind of result (more on that later). This goes for example from training a model with a given dataset, to obtaining image samples for certain attack.
Then, in order to run an experiment we first need to write a *tasks file*, which is a json file where we specify the different tasks to execute and all models, datasets, attacks and defenses to use.

A simple *tasks file* could look like this:
```
{
  "tasks": [
    {
      "task_data": {"task_name": "accuracy"},
      "nets": [
        {
          "model_name": "resnet18",
          "datasource_name": "imagenet",
          "net_id": "resnet18_imagenet"
        }
      ],
      "attacks": [{"attack_name": "fgsm"}],
      "defenses": [{"defense_name": "jpeg"}]
    }
  ]
}
```

In it we can identify 4 main components:
* ``task_data``: the actual ``Task`` component to use.
* ``nets``: specification of the models, datasets and model weights to use in the task's execution (respectively).
* ``attacks``: the set of attacks to use.
* ``defenses``: the set of defenses to use.

There are some extra arguments to modify how tasks get executed, and it's also possible to pass arguments to the each of the involved components. To see a full description of the *tasks file* schema and all possible arguments execute the ``help.py`` file.

Finally we need to execute ``run.py`` passing the path to the *tasks file* as only argument.
In this example, the accuracy would be obtained for all combinations of the given models, attacks and defenses, that is, for the model alone, for the defended model (without attack), for the attacked model (without defense) and lastly for the defended model using the attack.
Results are then stored inside the ``results`` folder.


Assuming your *tasks file* is located inside some ``data/executables`` folder:
```
python run.py data/executables/test.json
```

## Implementation

The functioning of this framework is based on 5 types of user implementable components, these are:
* ``Model``: the models to use, subclasses of ``nn.module`` as in any normal PyTorch implementation.
* ``DataSource``: a custom dataset implementation built over PyTorch's ``DataLoader``.
* ``Attack``: attack implementations that convert clean images into adversarial samples.
* ``Defense``: defense mechanisms to improve robustness against attacks (also subclass of ``nn.module``).
* ``Task``: tasks, as said previously, contain all code we would like to execute using the other components.

All of the added components should be placed in the corresponding folders inside ``src/custom`` and then added to the corresponding ``KNOWN_[type]`` dictionary at ``src/control.py``. When adding an entry to any of these dictionaries, the selected key is how you will later refer to the component in the *tasks file* and the value must be the dotted path to the component, relative to the ``src/custom`` folder.
For example, if we would add a ResNet model to the control file, it could look something like this:
```
KNOWN_MODELS = {
    'resnet18': 'models.resnet.ResNet18',
}
```

In the following sections we go a bit into detail with each of these types of components and how to implement them, with exception of the models since they don't have any particularity.
For examples, see the built in components.

### DataSource
These are high level representations of datasets. They provide other components an uniform way of accessing the data, some features like image normalization, and all the advantages derived from using ``torch.utils.data.DataLoader``, while still making available the underlying ``torch.utils.data.Dataset`` when needed.

To add your own datasource create a new class that inherits from ``src.base.datasource.DataSource`` and implements the methods ``get_dataset`` and ``get_classes``. ``get_dataset`` receives a boolean indicating whether it needs the training or validation set, and returns an ``torch.utils.data.Dataset`` instance. ``get_classes`` should return an array with the human readable labels of the dataset.
See [here](https://PyTorch.org/docs/stable/torchvision/datasets.html) for a list of already implemented datasets for image processing tasks.

Since it's likely your datasource will always use the same value for some parameters, like mean, standard deviation and name, we recommended overwriting the ``__init__`` method to fix this arguments and avoid the need of adding them in each *tasks file*. Note that the ``__init__`` method of the base class should always be called in order for ``normalize``, ``unnormalize`` and ``clamp`` to work.

### Attack
Attacks, as the name says, contain the logic of the different attack algorithms. They receive references to the model or defense being attacked (remember, both are instances of ``nn.module``) and the ``src.base.datasource.DataSource`` instances being used, and should provide a way for converting clean images into adversarial samples.

To add an attack simply subclass ``src.base.attack.Attack`` and implement the ``__call__`` method, which receives a set of images and their correct labels and returns the adversarial samples.
It's also a good idea to override the ``__init__`` method so that it receives the attack hyperparameters as arguments and if possible stores them as object attributes. This allows you to modify these parameters at runtime using the *tasks file* and evaluate how varying it's values would affect the task's results.

### Defense
Defenses contain code intended to improve models robustness against different attacks. Like the attacks, they receive the model (the one to defend) and DataSource instance being used but behave like models themselves.
We consider two types of defenses, model-agnostic ones, which basically apply some sort of pre-processing to the images to try to remove the perturbations and model-specific ones that seek robustness in a different way.

For both types of defenses users must subclass ``src.base.defense.Defense``, exposing the following two methods:
* ``forward``: receives the images and returns their estimated labels with the defense applied. This method needs to be overwritten only on model-specific defenses.
* ``process``: receives the images and returns their 'clean' versions. This method should be overwritten only on model-agnostic defenses and is then used by the base implementation of ``forward``.

### Task
Tasks are responsible for manipulating and obtaining results from the other components. User defined tasks must subclass ``src.base.task.Task`` and implement at least one of ``exec_task_simple`` and ``exec_task_multi`` methods, which represent the two existing modes of task execution.
* ``exec_task_simple`` receives a path string, the model or defense being use (remember defenses behave like models), the datasource, and potentially an attack instance. Tasks should return json serializable content that will then be written to one of the results files. The given path is useful for when you need to save content in other format.
Keep in mind that this method is the one executed by default.
* ``exec_task_multi`` similarly to ``exec_task_simple`` receives a path, model (in this case it's never a defense instance), datasource and iterables of attacks and defenses, and must return json serializable content as well. The purpose of this method is to give users control over all the attacks and defenses simultaneously, allowing them to generate aggregated results if needed. As said before, the ``exec_task_simple`` method is called by default when executing a task, to indicate that ``exec_task_multi`` must be used, set the ``exec_multi`` flag to true inside ``task_data`` in the *tasks file*.


## Authors

* **Camila Serena**
* **Diego Irigaray**


## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
