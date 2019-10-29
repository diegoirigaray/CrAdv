from jsonschema import validate
from jsonschema.exceptions import ValidationError


net_schema = {
    "type": "object",
    "title": "Specification of a model and datasource.",
    "required": [
        "model_name",
        "datasource_name",
        "net_id"
    ],
    "properties": {
        "model_name": {
            "type": "string",
            "title": "Name of the model.",
            "description": "Name used to reference the model in the `control.py` file."
        },
        "model_params": {
            "type": "object",
            "title": "Params passed to the specified model.",
            "description": "Dictionary passed as kwargs to the model on inicialization."
        },
        "datasource_name": {
            "type": "string",
            "title": "Name of the datasource.",
            "description": "Name used to reference the datasource in the `control.py` file."
        },
        "datasource_params": {
            "type": "object",
            "title": "Params passed to the specified datasource.",
            "description": "Dictionary passed as kwargs to the datasource on inicialization."
        },
        "net_id": {
            "type": "string",
            "title": "Model weights identifier.",
            "description": ("Identifier used to store and rerstore the weights of the specified model.\n"
                            "When training, a file named `[net_id].pth` is created to store the resulting weights.\n"
                            "The same `net_id` must be specified each time you wish to use the trained model.\n"
                            "This allows to train multiple instances of the same model.\n"
                            "If the model weights are obtained from another source, place them at `data/weights/`\n"
                            "on a '.pth' file named as the desired `net_id`.")
        },
    }
}


defense_schema = {
    "type": "object",
    "title": "Specification of a defense.",
    "required": ["defense_name"],
    "properties": {
        "defense_name": {
            "type": "string",
            "title": "Name of the defense.",
            "description": "Name used to reference the defense in the `control.py` file."
        },
        "defense_params": {
            "type": "object",
            "title": "Params passed to the specified defense.",
            "description": "Dictionary passed as kwargs to the defense on inicialization."
        }
    }
}


attack_schema = {
    "type": "object",
    "title": "Specification of an attack.",
    "required": ["attack_name"],
    "properties": {
        "attack_name": {
            "type": "string",
            "title": "Name of the attack.",
            "description": "Name used to reference the defense in the `control.py` file."
        },
        "attack_params": {
            "type": "object",
            "title": "Params passed to the specified attack.",
            "description": "Dictionary passed as kwargs to the attack on inicialization."
        },
        "on_task_model": {
            "type": "boolean",
            "title": "Execute the attack using the task's model.",
            "description": ("Flag to enable usage of the task's model on the attack. If True, the attack gets executed\n"
                            "using the same model currently being used by the task (white-box approach), otherwise attacks\n"
                            "only using the models specified in `specific_models` (black-box aproach).\n"
                            "Default True.\n")
        },
        "specific_models": {
            "type": "array",
            "title": "Models to use on the attack",
            "description": ("If specified, an instance of the task will be executed with the attack using each\n"
                            "of the the given models, allowing execution in a black-box manner."),
            "items": net_schema
        },
        "except_variables": {
            "type": "array",
            "title": "List of attack variables to ignore in this attack",
            "description": ("If any of the attack variables specified with `attack_variables` shouldn't be used with this\n"
                            "attack, it's name (`variable_name`) can be added to this array to avoid it's execution."),
            "items": {
                "type": "string",
                "title": "Name of the variable (`variable_name`) to be ignored."
            }
        }
    }
}


attack_variable = {
    "type": "object",
    "title": "Testing values for an attack hyperparameter.",
    "description": ("Name and values for attack's hyperparameter on which the given task will be executed.\n"
                    "The specified variables will be set onto the attacks using their `set_attr` method.\n"
                    "The results rely on the attack's implementation actually using the specified parameter.\n"
                    "Attack variables are usefull to test (and potentially plot) the effect of certain attack\n"
                    "hyperparameters over a given task."),
    "required": ["variable_name", "variable_values"],
    "properties": {
        "variable_name": {
            "type": "string",
            "title": "Name of the hyperparameter."
        },
        "variable_values": {
            "type": "array",
            "title": "List of values for the specified hyperparameter.",
            "items": {
                "title": "Values for the specified variable."
            }
        }
    }
}


task_data_schema = {
    "type": "object",
    "title": "Specification of the actual `Task`.",
    "required": ["task_name"],
    "properties": {
        "task_name": {
            "type": "string",
            "title": "Name of the task.",
            "description": "Name used to reference the task in the `control.py` file."
        },
        "task_params": {
            "type": "object",
            "title": "Params passed to the specified task.",
            "description": "Dictionary passed as kwargs to the task on inicialization."
        },
        "exec_multi": {
            "type": "boolean",
            "title": "Flag to switch between task's execution modes.",
            "description": ("By default, tasks are executed for each defense and attack, one combination at a time.\n"
                            "Set this flag to True to pass iterables of every defense and attack to the task. Keep in mind\n"
                            "that the task must implement the `exec_task_multi` method.")
        },
        "attack_on_defense": {
            "type": "boolean",
            "title": "Flag to select which component gets passed to attacks.",
            "description": ("When evaluating attacks and defenses simultaneously, attacks may receive either the undefended\n"
                            "model or the defense instance. This flag let's you select which one get's passed.\n"
                            "This is usefull when working with non differentiable defenses or assuming an atacker with\n"
                            "no defense knowledge. By default this flag is set to `true` and uses the defense.")
        },
        "plot_keys": {
            "type": "array",
            "title": "Allows to plot the specified properties of the task's result.",
            "description": ("When executing a task (that returns a dict) using `attack_variables`, specify here the properties\n"
                            "of the result you wish to plot. Specified properties must contain a single number."),
            "items": {"type": "string"}
        },
        "plot_together": {
            "type": "boolean",
            "title": "Flag to specify if the specified keys should be plotted together.",
            "description": ("If true, and multiple keys were specified in `plot_keys`, the different properties\n"
                            "will be drawn in the same plot. Default is true.")
        },
        "skip_no_defense": {
            "type": "boolean",
            "title": "Flag to enable/disable execution of a task without defenses.",
            "description": ("By default, tasks are executed for each defense and for the undefended model.\n"
                            "To disable the execution of the task on the undefended model, set this flag to True.")
        },
        "skip_no_attack": {
            "type": "boolean",
            "title": "Flag to enable/disable execution of a task without attacks.",
            "description": ("By default, tasks are executed without attacks and then for each specified attack.\n"
                            "To disable the execution of the task without attacks, set this flag to True.")
        },
        "skip_no_attack_variables": {
            "type": "boolean",
            "title": "Flag to enable/disable execution of an attack without the attack variables.",
            "description": ("By default, attacks are used first with their inicialization parameters and then for each of the specified\n"
                            "attack variables. To disable the execution of the task using the inicialization parameters, set this flag to False.")
        },
    }
}


task_schema = {
    "type": "object",
    "title": "Specification for a task execution.",
    "description": ("Full specification of an executable task. It includes the information of the task it self along with\n"
                    "the models, datasources, defeses and attacks with which the task will be executed.\n"
                    "The `Task` specified in `task_data` gets executed with all the posible combinations of models, defenses\n"
                    "and attacks given (in addition to the attack variables)."),
    "required": ["task_data", "nets"],
    "properties": {
        "task_data": task_data_schema,
        "attack_variables": {
            "type": "array",
            "title": "Attack hyperparameters to evaluate on the task.",
            "items": attack_variable
        },
        "nets": {
            "type": "array",
            "title": "Models and datasources to use on the task.",
            "items": net_schema
        },
        "defenses": {
            "type": "array",
            "title": "Defenses to use on the task.",
            "items": defense_schema
        },
        "attacks": {
            "type": "array",
            "title": "Attacks to use on the task.",
            "items": attack_schema
        }
    }
}


config_schema = {
    "type": "object",
    "title": "Configuration for the tasks execution.",
    "properties": {
        "device": {
            "type": "string",
            "title": "torch.device to use in the tasks execution.",
            "description": "If device is not specified, uses 'cuda:0' when there is an available device, otherwise uses 'cpu'."
        },
        "multi_gpu": {
            "type": "boolean",
            "title": "Flag to enable/disable usage of multiple gpus.",
            "description": "By default, uses all available gpus. Set this flag to False to use one gpu max."
        },
        "results_path": {
            "type": "string",
            "title": "Path on where to store the execution results.",
            "description": "By default stores the results in the 'results' folder, in a sub-folder named with the date and time."
        },
        "device_ids": {
            "type": "array",
            "title": "CUDA devices to use.",
            "description": ("List of int or torch.device passed to DataParallel when `multi_gpu` is enabled.\n"
                            "Allows to specify which of all available devices to use."),
            "items": {
                "type": "string",
                "title": "Device identifier."
            }
        },
        "batch_size_factor": {
            "type": "number",
            "title": "Factor to adapt the batch size of this experiment.",
            "description": ("If you need to execute a tasks file on a different machine with more/less memory you can\n"
                            "use this parameter to increase/descrease the batch size used for all datasources, instead of\n"
                            "modifying all appearences in the tasks file.\n")
        },
        "safe_mode": {
            "type": "boolean",
            "title": "Flag to enable/disable the propagation of exceptions.",
            "description": ("By default, exceptions are propagated up and may cause the whole execution to stop.\n"
                            "Setting this flag to True causes the scheduler to catch any exceptions raised during task execution,\n"
                            "allowing the execution of other tasks to continue. Catched exceptions are logged\n"
                            "in the corresponding results file.")
        }
    }
}

schedule_schema = {
    "type": "object",
    "title": "Specification of all tasks to be executed.",
    "required": ["tasks"],
    "properties": {
        "config": config_schema,
        "tasks": {
            "type": "array",
            "title": "List of tasks.",
            "items": task_schema
        }
    }
}


def validate_schedule(json):
    validate(json, schedule_schema)
