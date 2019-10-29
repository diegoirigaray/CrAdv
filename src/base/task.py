

class Task(object):
    '''
    Base class for all executable tasks.

    Your tasks must subclass this class.

    Tasks are ment to implement logic you wish to use to test combinations of models,
    datasources, attacks and defenses (like the accuracy of a network), or basically
    any 'task' you would like to do with them (like network training).

    Tasks expose three methods that are used by `src.exec.scheduler.Scheduler`:
    `exec_task_simple` which executes the task's logic for a given model/defense,
    datasource and attack, `exec_attack_eval` that basically calls `exec_task_simple`
    varying attack's parameters between calls, and `exec_task_multi`, that allows to get
    aggregates results by using multiple attacks and defenses.

    Custom tasks must implement at least one of `exec_task_simple` or `exec_task_multi`
    methods. Keep in mind that by default, the scheduler will call `exec_task_simple`,
    and in order to use `exec_task_multi` you must set the task's file `exec_task_multi`
    flag to True.
    '''
    def exec_attack_eval(self, results_path, net, datasource, attack, var_name,
                         var_values):
        '''
        Executes the task multiple times changing the attack configuration.

        Uses the attack's `set_attr` method to modify some `var_name` attribute of the
        attack with the given `var_values` values, calling `exec_task_simple` each time.
        Useful for testing how certain parameter affects an attack behavior.

        Args:
            net (nn.Module or base.defense.Defense): model or defense used on the
                current execution.
            datasource (base.datasource.DataSource): DataSource, normally the one on
                which the given model is trained.
            attack (base.attack.Attack): attack of the current execution.
            var_name (string): name of the attack's attribute to be modified
                between task's executions.
            var_values (list): list of the values to be set to `var_name` on each
                execution.

        Returns:
            dict: dictionary containing the used variable name and a results dictionary
                with the `var_values` elements as keys and the corresponding
                `exec_task_simple` result as value.
        '''
        results = {}
        for val in var_values:
            attack.set_attr(var_name, val)
            results[val] = self.exec_task_simple(results_path, net, datasource, attack)

        result = {
            'variable_name': var_name,
            'results': results
        }
        return result


    def exec_task_simple(self, results_path, net, datasource, attack=None):
        """
        Executes the task's logic for a given model/defense, datasource and attack.

        Implement this method if you wish to use your task with one defense and attack
        at a time. Keep in mind this is the default mode of execution for tasks.

        The result of this function must be json serializable in order to be written in
        the result's file.
        If you need to save non serializable content use the 'results_path' argument,
        which points to a folder created for this current execution.

        Args:
            results_path (string): path to a folder located where the task's result will
                be saved. Use this path if you need to save non serializable content.
            net (nn.Module or base.defense.Defense): model or defense used on the
                current execution. Keep in mind that the implemented defenses may not be
                differentiable.
            datasource (lid.datasource.DataSource): datasource, normally the one on
                which the given model is trained.
            attack (base.attack.Attack, optional): attack of the current execution.

        Returns:
            dict: Result of the task for the given configuration. Must be json
                serializable. Its written by `src.scheduler.Scheduler` in the
                corresponding result's file.
        """
        raise NotImplementedError

    def exec_task_multi(self, writer, results_path, net, datasource, attacks, defenses):
        """
        Executes the task's logic for a set of attacks and defenses.

        Similar to the `exec_task_simple` method, but in this case it receives the model,
        the datasource, and iterables of all specified attacks and defenses.
        Implement this method if you wish to generate an aggregated result from all
        specified attacks and defenses.

        Note: the first elements of the `attacks` and `defenses` iterables may be `None`
              (when `skip_no_defense` and `skip_no_attack` are False) which means
              that the task should be executed without attack/defense in that iteration.

              Also, when running on this mode the attacks iterable uses the model
              instance regardless of the value of the `attack_on_defense` flag. Use the
              `set_defense` method of the attacks iterable if you wish to pass the
              defense to the attacks.

        Args:
            writer (exec.writer.Writer): instance of the Writer if the task needs to
                save results or get a results folder manually.
            results_path (string): path on where the task's result will be saved. Use
                this path if you need to save non serializable content.
            net (nn.Module): model used on the current execution.
            datasource (lid.datasource.DataSource): datasource, normally the one on
                which the given model is trained.
            attacks (iterable of base.attack.Attack/None): attacks for the current
                execution, one of the elements may be None.
            defenses (iterable of base.defense.Defense/None): defenses of the current
                execution, one of the elements may be None.

        Returns:
            object: Result of the task for the given configuration. Must be json
                serializable. Its written by `src.scheduler.Scheduler` in the
                corresponding result's file.
        """
        raise NotImplementedError
