import torch
import math
from matplotlib import pyplot as plt
from src.base import Task
from src.utils.functions import rescale_adversarial


class ConstrainedAccuracy(Task):
    '''
    Task to get the accuracy of a given net by constraining the perturbation norm.

    Args:
        limit_samples (int, optional): if given, limits the amount of samples used
            to get the accuracy, by default uses all the testing set is.
    '''
    def __init__(self, limit_samples=None, top_k=None, dissimilarity=True,
                 contraint_step=0.25, constraint_min=0, constraint_max=6, **kwargs):
        self.limit_samples = limit_samples
        self.top_k = top_k
        self.dissimilarity = dissimilarity
        self.contraint_step = contraint_step
        self.constraint_min = constraint_min
        self.constraint_max = constraint_max

    def process(self, net, datasource, attack, percentage_factor, current=None):
        attr = "dissimilarity" if self.dissimilarity else "max_2_norm"
        total = 0
        info = {}
        print('Completed {}: {:.2f} %'.format(current, total*percentage_factor), end='\r')

        val = self.constraint_min
        while val <= self.constraint_max:
            info[val] = {'correct': 0, 'adv_dissimilarity': 0}
            if self.top_k:
                info[val]['correct_topk'] = 0
            val += self.contraint_step

        with torch.no_grad():
            for data in datasource.get_dataloader(train=False):
                o_images, labels = data
                total += labels.size(0)

                u_images = torch.stack([datasource.unnormalize(i) for i in o_images])
                norm_2_clean = torch.norm(torch.flatten(u_images, start_dim=1), dim=1)

                if not hasattr(attack, attr):
                    adversarial = attack(o_images, labels)

                # Rescale the perturbations to get constrained data
                for val, data in info.items():
                    if val == 0:
                        scaled = o_images
                    else:
                        if hasattr(attack, attr):
                            # When the attack accepts a norm attribute, uses it to get
                            # a good attack for the given norm
                            attack.set_attr(attr, val)
                            adversarial = attack(o_images, labels)

                        # Otherwise, gets the attack without norm restriction and
                        # rescales it to the desired norm
                        if self.dissimilarity:
                            scaled = rescale_adversarial(
                                adversarial, u_images, norm_2_clean, datasource, val)
                        else:
                            scaled = rescale_adversarial(
                                adversarial, u_images, norm_2_clean, datasource, norm=val)

                    outputs = net(scaled)

                    # Adds the correct predictions count
                    _, predicted = torch.max(outputs.data, 1)
                    data['correct'] += (predicted == labels).sum().item()

                    # Adds the batch adversarial dissimilarity
                    u_scaled = torch.stack([datasource.unnormalize(i) for i in scaled])
                    norm_perts = torch.norm(torch.flatten(u_scaled - u_images, start_dim=1), dim=1)
                    data['adv_dissimilarity'] += torch.sum(norm_perts / norm_2_clean).item()

                    if self.top_k:
                        _, top_predictions = torch.topk(outputs, self.top_k)
                        for i, top in enumerate(top_predictions):
                            data['correct_topk'] += int(labels[i].item() in top)

                print('Completed {}: {:.2f} %'.format(current, total*percentage_factor),
                                                      end='\r')
                if self.limit_samples and total >= self.limit_samples:
                    break

        # Adds the percentage value
        for val, data in info.items():
            data['accuracy'] = 100.0 * data['correct'] / total
            data['adv_dissimilarity'] = data['adv_dissimilarity'] / total
            if self.top_k:
                data['topk_accuracy'] = 100.0 * data['correct_topk'] / total

        return info

    def exec_task_simple(self, results_path, net, datasource, attack):
        limit = self.limit_samples or datasource.get_dataset(False).__len__()
        percentage_factor = 100 / limit
        total = math.ceil(limit / datasource.batch_size) * datasource.batch_size

        info = self.process(net, datasource, attack, percentage_factor)
        print()

        # Plot the results
        x_values = [key for key, _ in info.items()]
        y_values = [val['accuracy'] for _, val in info.items()]
        plt.xlabel("Norm")
        plt.ylabel("Accuracy")
        plt.plot(x_values, y_values, label="Accuracy")
        if self.top_k:
            y_values = [val['topk_accuracy'] for _, val in info.items()]
            plt.plot(x_values, y_values, label="Accuracy top {}".format(self.top_k))
        plt.legend()
        plt.savefig('{}/accuracy_plot.png'.format(results_path))
        plt.close()

        result = {}
        result['total'] = total
        result['accuracy_by_norm'] = info

        print('\n')
        return result

    def exec_task_multi(self, writer, results_path, net, datasource, attacks, defenses):
        limit = self.limit_samples or datasource.get_dataset(False).__len__()
        percentage_factor = 100 / limit
        total = math.ceil(limit / datasource.batch_size) * datasource.batch_size
        combinations = defenses.count() * attacks.count()
        current = 1

        info = {}
        for defense in defenses:
            defense_info = {}
            if defense and self.data['attack_on_defense']:
                attacks.set_defense(defense)
            for attack in attacks:
                if not attack:
                    continue
                defense_info[attack.data['attack_name']] = self.process(
                    defense if defense else net, datasource, attack, percentage_factor,
                    "({}/{})".format(current, combinations))
                current += 1

            info[defense.data['defense_name'] if defense else 'no_defense'] = defense_info

        print()

        # Plot the results
        for defense, d_data in info.items():
            plt.xlabel("Norm")
            plt.ylabel("Accuracy ({})".format(defense))

            for attack, a_data in d_data.items():
                x_values = [key for key, _ in a_data.items()]
                y_values = [val['accuracy'] for _, val in a_data.items()]
                plt.plot(x_values, y_values, label=attack)

            plt.legend()
            plt.savefig('{}/{}_plot.png'.format(results_path, defense))
            plt.close()

        if self.top_k:
            for defense, d_data in info.items():
                plt.xlabel("Normalized L2 dissimilarity")
                plt.ylabel("Accuracy top {} ({})".format(self.top_k, defense))

                for attack, a_data in d_data.items():
                    x_values = [key for key, _ in a_data.items()]
                    y_values = [val['topk_accuracy'] for _, val in a_data.items()]
                    plt.plot(x_values, y_values, label=attack)

                plt.legend()
                plt.savefig('{}/{}_topk_plot.png'.format(results_path, defense))
                plt.close()

        result = {}
        result['total'] = total
        result['accuracy_by_norm'] = info

        print('\n')
        return result
