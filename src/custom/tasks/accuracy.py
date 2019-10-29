
import torch
from src.base import Task
from src.utils.functions import to_percentage


class Accuracy(Task):
    '''
    Task to get the accuracy of a given net.

    Args:
        limit_samples (int, optional): if given, limits the amount of samples used
            to get the accuracy, by default uses all the testing set.
        top_k (int, optional): if given, this task also calculates the top `top_k`
            accuracy.
    '''
    def __init__(self, limit_samples=None, top_k=None, round=False, **kwargs):
        self.limit_samples = limit_samples
        self.top_k = top_k
        self.round = round

    def exec_task_simple(self, results_path, net, datasource, attack=None):
        total = 0
        correct = 0
        correct_topk = 0
        adv_count = 0
        c_total = 0

        limit = self.limit_samples or datasource.get_dataset(False).__len__()
        percentage_factor = 100 / limit
        print('Completed: {0:.2f} %'.format(total*percentage_factor), end='\r')

        # Dataset norms
        image = datasource.get_dataloader(train=False).__iter__().next()[0][0]
        norm_0_ds = image.numel()
        norm_2_ds = 0
        norm_inf_ds = 0
        clean_conf = 0
        # Perturbation norms
        norm_0_adv = 0
        norm_2_adv = 0
        adv_dissimilarity = 0
        adv_inf_dissimilarity = 0
        norm_inf_adv = 0
        adv_conf = 0

        with torch.no_grad():
            for data in datasource.get_dataloader(train=False):
                o_images, labels = data
                total += labels.size(0)

                # Adds the images norms, to get the dataset avg norms
                u_o_images = torch.stack([datasource.unnormalize(i) for i in o_images])
                flat = torch.flatten(u_o_images, start_dim=1)
                norm_2_clean = torch.norm(flat, dim=1)
                norm_inf_clean = torch.norm(flat, float('inf'), 1)
                norm_2_ds += torch.sum(norm_2_clean).item()
                norm_inf_ds += torch.sum(torch.norm(flat, p=float('inf'), dim=1)).item()

                # Use the attack when given
                if attack:
                    images = attack(o_images, labels)
                    if self.round:
                        images = datasource.round_pixels_batch(images)
                else:
                    images = o_images
                outputs = net(images)

                # Adds the correct predictions count
                _, predicted = torch.max(outputs.data, 1)
                confidence, _ = torch.max(torch.nn.functional.softmax(outputs, 1), 1)
                w_correct = predicted == labels
                correct += w_correct.sum().item()

                # Confidence of correct classification
                c_positions = (w_correct == 1).nonzero().flatten()
                clean_conf += torch.index_select(confidence, 0,
                                                         c_positions).sum().item()

                if attack:
                    # Gets the perturbations that fooled the net
                    positions = (w_correct == 0).nonzero().flatten()
                    if len(positions) != 0:
                        # Discard the ones that were originally missclasified
                        fooled = torch.index_select(o_images, 0, positions)
                        fooled_labels = torch.index_select(labels, 0, positions)
                        f_output = net(fooled)
                        _, pre_predictions = torch.max(f_output.data, 1)
                        positions = (pre_predictions == fooled_labels).nonzero().flatten()

                        c_total += len(labels) - len((
                            pre_predictions != fooled_labels).nonzero().flatten())

                        # For the perturbations norms avg only consider the samples
                        # that fooled the net but were originally classified correctly
                        if len(positions) != 0:
                            adv_count += len(positions)
                            adv_images = torch.index_select(images, 0, positions)
                            u_images = torch.stack([datasource.unnormalize(i) for i in adv_images])
                            u_o_images = torch.index_select(u_o_images, 0, positions)
                            fool_perts = u_images - u_o_images

                            # Add the fooled examples confidence
                            adv_conf += torch.index_select(confidence, 0,
                                                           positions).sum().item()

                            # Adds the norms
                            flat = torch.flatten(fool_perts, start_dim=1)
                            norm_0_adv += torch.sum(torch.norm(flat, 0, 1)).item()
                            norm_2_att = torch.norm(flat, dim=1)
                            norm_2_adv += torch.sum(norm_2_att).item()
                            adv_dissimilarity += torch.sum(
                                norm_2_att / torch.index_select(norm_2_clean, 0, positions)).item()
                            norm_inf_adv += torch.sum(
                                torch.norm(flat, float('inf'), 1)).item()
                            adv_inf_dissimilarity += torch.sum(
                                torch.norm(flat, float('inf'), 1) / torch.index_select(norm_inf_clean, 0, positions)).item()

                if self.top_k:
                    _, top_predictions = torch.topk(outputs, self.top_k)
                    for i, top in enumerate(top_predictions):
                        correct_topk += int(labels[i].item() in top)

                print('Completed: {0:.2f} %'.format(total*percentage_factor), end='\r')
                if self.limit_samples and total >= self.limit_samples:
                    break

        print()


        result = {}
        accuracy = correct / total
        result['total'] = total
        result['correct'] = correct
        result['adversarial'] = adv_count
        result['accuracy'] = accuracy

        result['avg_confidence'] = {}
        result['avg_confidence']['correct'] = (clean_conf / correct) if correct else "-"
        print('Accuracy: {}'.format(to_percentage(accuracy)))

        if self.top_k:
            top_k_accuracy = correct_topk / total
            print('Accuracy (Top {}): {}'.format(
                self.top_k, to_percentage(top_k_accuracy)))
            result['top_{}_accuracy'.format(self.top_k)] = top_k_accuracy

        if attack:
            c_accuracy = (c_total - adv_count) / c_total if c_total else 1
            result['clean_correct'] = {}
            result['clean_correct']['total'] = c_total
            result['clean_correct']['adversarial'] = adv_count
            result['clean_correct']['accuracy'] = c_accuracy
            result['avg_confidence']['adversarial'] = (adv_conf / adv_count) if adv_count else "-"

            adv_count = 1 if adv_count == 0 else adv_count
            result['clean_avg_norm'] = {}
            result['clean_avg_norm']['0'] = norm_0_ds
            result['clean_avg_norm']['2'] = norm_2_ds / total
            result['clean_avg_norm']['inf'] = norm_inf_ds / total

            result['adv_avg_norm'] = {}
            result['adv_avg_norm']['0'] = norm_0_adv / adv_count
            result['adv_avg_norm']['2'] = norm_2_adv / adv_count
            result['adv_avg_norm']['inf'] = norm_inf_adv / adv_count

            result['adv_dissimilarity'] = {}
            result['adv_dissimilarity']['2'] = adv_dissimilarity / adv_count
            result['adv_dissimilarity']['inf'] = adv_inf_dissimilarity / adv_count

        print('\n')
        return result
