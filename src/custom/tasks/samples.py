import torch
from torchvision.utils import save_image
from src.base import Task, Defense
from src.utils.functions import to_percentage


class Samples(Task):
    '''
    Task to get image samples of an experiment.

    Saves image samples of a dataset, potentially applying a defense and attack to them.

    Args:
        limit_samples (int, optional): max amount of samples saved for each of the
            possible cases.
        save_clean (bool, optional): flag that enables saving clean images.
        save_defended (bool, optional): flag that enables saving images with the defense
            applied (only when using model-agnostic defenses with `process` implemented).
        save_adversary (bool, optional): flag that enables saving images with an attack
            applied.
        save_perturbation (bool, optional): flag that enables saving images of the
            perturbations.

    Note: when both attack and defense are given, the defense will be applied over
        the already adversarial image.
    '''
    def __init__(self, limit_samples=4, save_clean=True, save_defended=True,
                 save_adversary=True, save_perturbation=True, **kwargs):
        super().__init__(**kwargs)
        self.limit_samples = max(limit_samples, 1)
        self.save_clean = save_clean
        self.save_defended = save_defended
        self.save_adversary = save_adversary
        self.save_perturbation = save_perturbation

        # The samples dict is used to temporaly store images before saving them to disk.
        self.samples = {}

    def process(self, net, datasource, images, labels, index, results_path, attack):
        samples = []
        correct = 0
        is_defense = isinstance(net, Defense)
        c_outputs = net(images)
        clean_conf, _ = torch.max(torch.nn.functional.softmax(c_outputs, 1), 1)

        # Saves the clean images
        for i, img in enumerate(images):
            samples.append({
                "index": index + i + 1,
                "original_label": self.classes[labels[i].item()],
                "clean_confidence": to_percentage(clean_conf[i].item())})

            if self.save_clean:
                # Unnormalizes the image before saving it
                clean_img = datasource.unnormalize(img)
                name = "{}_clean".format(index + i + 1)
                save_image(clean_img, "{}/{}.png".format(results_path, name))

        # Saves images after modified by the attack
        if attack:
            adversarial = attack(images, labels)
            outputs = net(adversarial)
            _, predicted = torch.max(outputs.data, 1)
            adv_conf, _ = torch.max(torch.nn.functional.softmax(outputs, 1), 1)
            correct += (predicted == labels).sum().item()

            for i, img in enumerate(adversarial):
                samples[i]["adversary_label"] = self.classes[
                    predicted[i].item()]

                # Adds the norms of the perturbation to the results
                pert = torch.add(datasource.unnormalize(img), -1,
                                 datasource.unnormalize(images[i]))
                flat = torch.flatten(pert)
                samples[i].update({
                    "adversary_label": self.classes[predicted[i].item()],
                    "adversary_confidence": to_percentage(adv_conf[i].item()),
                    "pert_norm_0": torch.norm(pert, 0).item(),
                    "pert_norm_2": torch.norm(pert).item(),
                    "pert_norm_inf": torch.norm(pert, float('inf')).item()
                })

                # Saves the adversarial image
                if self.save_adversary:
                    adv_img = datasource.unnormalize(img)
                    a_name = "{}_adversary".format(index + i + 1)
                    save_image(adv_img, "{}/{}.png".format(results_path, a_name))

                # Saves the perturbation vector
                if self.save_perturbation:
                    # Scales the perturbation to the complete pixel range
                    flat = torch.flatten(pert, start_dim=1)
                    scaled = flat / torch.reshape(
                        torch.norm(flat, float('inf'), 1), [len(pert), 1])
                    scaled = torch.reshape(scaled, pert.size())
                    p_name = "{}_perturbation".format(index + i + 1)
                    save_image(scaled, "{}/{}.png".format(results_path, p_name))

        # Stores images after the defense processing
        if is_defense and self.save_defended:
            images = net.process(images)
            for i, img in enumerate(images):
                datasource.unnormalize(img, True)
                name = "{}_defended".format(index + i + 1)
                save_image(img, "{}/{}.png".format(results_path, name))

        results = {
            'samples': samples,
            'correct': correct
        }
        return results

    def exec_task_simple(self, results_path, net, datasource, attack=None):
        total = 0
        correct = 0
        samples = []
        self.classes = datasource.get_classes()
        percentage_factor = 100 / self.limit_samples if self.limit_samples else None

        with torch.no_grad():
            for data in datasource.get_dataloader(train=False):
                images, labels = data

                result = self.process(net, datasource, images, labels, total,
                                      results_path, attack)

                total += labels.size(0)
                correct += result['correct']
                samples.extend(result['samples'])

                if not self.limit_samples:
                    print('Total: {}'.format(total), end='\r')
                else:
                    print(
                        'Completed: {0:.2f} %'.format(total*percentage_factor), end='\r')
                    if total >= self.limit_samples:
                        break
        print()

        results = {}
        results['total'] = total
        results['correct'] = correct
        results['samples'] = samples
        return results

    def exec_task_multi(self, writer, results_path, net, datasource, attacks, defenses):
        results = {}
        ids = []
        paths = []
        total = 0
        self.classes = datasource.get_classes()
        percentage_factor = 100 / self.limit_samples if self.limit_samples else None

        for defense in defenses:
            ids_def = []
            paths_def = []
            for attack in attacks:
                path = writer.get_task_path(net, self, defense, attack)
                id = path.split("/")[-2]
                paths_def.append(path)
                ids_def.append(id)
                results[id] = {'correct': 0, 'samples': []}
            paths.append(paths_def)
            ids.append(ids_def)

        with torch.no_grad():
            for data in datasource.get_dataloader(train=False):
                images, labels = data

                for i, defense in enumerate(defenses):
                    for j, attack in enumerate(attacks):
                        id = ids[i][j]
                        path = paths[i][j]

                        result = self.process(defense if defense else net, datasource,
                                              images.clone(), labels, total, path, attack)


                        results[id]['correct'] += result['correct']
                        results[id]['samples'].extend(result['samples'])

                total += labels.size(0)
                if not self.limit_samples:
                    print('Total: {}'.format(total), end='\r')
                else:
                    print(
                        'Completed: {0:.2f} %'.format(total*percentage_factor), end='\r')
                    if total >= self.limit_samples:
                        break
        print()
        return results
