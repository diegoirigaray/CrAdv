import json
import torch
import torch.nn as nn

from src.control import Control
from src.utils.functions import save_weights, Timer
from src.utils.schemas import validate_schedule
from .writer import Writer
from .parser import Parser, DefenseIterable, AttackIterable, filter_attack_variables


RESULTS_BASE_PATH = 'results'


class Scheduler(object):
    """
    Class responsable for executing the tasks file.

    Executes each task in the tasks file with the specified models, datasources,
    defenses and attacks, all of which must be registered in `control.py`.
    For a specification of the expected format see `schemas.py` in the utils folder or
    execute the `help.py` file.

    All components (tasks it self, models, defenses...) are parsed using the `Parser`
    class, which sets defaults values and performs some extra work, like instantiating
    classes, and adding logging data.
    """

    def __init__(self):
        self.parser = Parser()
        self.timer = Timer()

    def __call__(self, tasks_file):
        """
        Opens and executes the specified tasks file.

        Loads the specified json file, reads the configuration info if given and then
        calls the `task_handler` for each one of the tasks.

        Args:
            tasks_file (string): path to the tasks file, with a valid `schedule_schema`
                from `schemas.py`
        """
        with open(tasks_file) as f:
            # Loads and validates the execution file
            schedule = json.load(f)
            validate_schedule(schedule)
            self.config = self.parser.parse_config(schedule.get("config", {}))
            self.writer = Writer(self.config['results_path'])

            if self.config["multi_gpu"]:
                if self.config['device_ids'] is not None:
                    total_gpus = len(self.config['device_ids'])
                else:
                    total_gpus = torch.cuda.device_count()
                print("**Using {} GPUs**".format(total_gpus))
            else:
                print("**Using {}**".format(self.config["device"]))

            # Executes all tasks in the schedule file
            for task in schedule['tasks']:
                self.task_handler(task)

    def task_handler(self, task_info):
        '''
        Parses and executes a single task.

        Instantiates the actual task specified in `task_info.task_data` and
        executes it for each of the specified models.
        Applies the specified defenses and attacks depending on the values of the flags
        `with_****` and `without_****` in `task_info.task_data`.

        Args:
            task_info (dict): specification of a task with a valid `task_schema`
                of the `schemas.py` file.
        '''
        task = self.parser.parse_task(task_info['task_data'])
        attack_vars = task_info.get("attack_variables", [])

        # Iterates over each net
        for net_item in task_info['nets']:
            net, datasource = self.parser.parse_net(net_item, self.config)
            print("Execution of task: <{}> with net: <{}> on datasource: <{}>".format(
                task.data['task_name'], net_item['model_name'],
                net_item['datasource_name']))

            # Gets the defenses iterable
            defenses = DefenseIterable(task.data['skip_no_defense'], net, datasource,
                                       task_info.get('defenses', []))
            # If 'attack_on_defense' is false or 'multi' execution is enabled,
            # gets the attack iterables using the current model
            if not task.data['attack_on_defense'] or task.data['exec_multi']:
                attacks = AttackIterable(task.data['skip_no_attack'], net, datasource,
                                         task_info.get('attacks', []), self.config)

            # If task 'multi' execution mode is enabled calls `exec_task_multi`
            if task.data['exec_multi']:
                task_path = self.writer.get_task_path(net, task)
                self.timer.start()
                try:
                    result = task.exec_task_multi(
                        self.writer, task_path, net, datasource, attacks, defenses)
                except Exception as e:
                    if self.config["safe_mode"]:
                        result = e
                    else:
                        raise e
                self.writer.save_results(task, net, task_info.get('defenses'),
                                         task_info.get('attacks'), result,
                                         self.timer.stop())
                continue

            # Else, calls `exec_task_simple` for each combination of attack/defense
            for defense in defenses:
                # If `attack_on_defense`, passes the current defense to the attacks
                if task.data['attack_on_defense']:
                    attacks = AttackIterable(task.data['skip_no_attack'], defense or net,
                                             datasource, task_info.get('attacks', []),
                                             self.config)
                for attack in attacks:
                    task_path = self.writer.get_task_path(net, task, defense, attack)

                    # Executes the task for the given configuration
                    if not attack or not task.data['skip_no_attack_variables']:
                        self.timer.start()
                        try:
                            result = task.exec_task_simple(
                                task_path, defense if defense else net, datasource,
                                attack)
                        except Exception as e:
                            if self.config["safe_mode"]:
                                result = e
                            else:
                                raise e
                        self.writer.save_results(task, net, defense, attack, result,
                                                 self.timer.stop())

                    # Executes the task for a set of attack variables
                    for v_name, v_val in filter_attack_variables(attack, attack_vars):
                        print("**Evaluating over variable: <{}>**".format(v_name))
                        self.timer.start()
                        try:
                            result = task.exec_attack_eval(
                                task_path, net, datasource, attack, v_name, v_val)
                        except Exception as e:
                            if self.config["safe_mode"]:
                                result = e
                            else:
                                raise e
                        self.writer.save_results(task, net, defense, attack, result,
                                                 self.timer.stop())
