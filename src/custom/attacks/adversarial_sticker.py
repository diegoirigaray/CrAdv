import math
import torch
import random
from PIL import Image
from src.base import Attack
from torch.nn import CrossEntropyLoss
from torch.nn.functional import interpolate, pad, softmax
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, resize


S_SIZE = 300  # True size of the resulting sticker
I_SIZE = 224  # Size of the dataset imamges


class AdversarialSticker(Attack):
    '''
    Custom implementation of the adversarial patch concept
    '''
    def __init__(self, model, datasource, base_img_path=None, adv_sticker_path=None, limit_samples=1000,
                 transforms_per_img=6, iter_per_img=10, adv_target=None, alpha=0.015, decay_factor=1,
                 gradient_noise=0.15, random_noise=0.15, save_path="adv_sticker.png", **kwargs):
        super().__init__(model, datasource, **kwargs)
        self.limit_samples = limit_samples
        self.transforms_per_img = transforms_per_img
        self.iter_per_img = iter_per_img
        self.adv_target = adv_target
        self.alpha = alpha
        self.decay_factor = decay_factor
        self.gradient_noise = decay_factor
        self.random_noise = random_noise
        self.save_path = save_path

        # If adversarial sticker already created, uses it
        if adv_sticker_path:
            sticker = Image.open(adv_sticker_path)
            self.adv_sticker = self.mask_circle(self.datasource.normalize(to_tensor(sticker))).to(self.datasource.device)
            return

        # If `base_img_path` is given, uses it as base image, otherwise starts from black img
        if not base_img_path:
            base_img = self.datasource.normalize(torch.zeros(3, S_SIZE, S_SIZE)).to(self.datasource.device)
        else:
            base_img = Image.open(base_img_path)
            base_img = resize(base_img, [S_SIZE, S_SIZE])
            base_img = self.datasource.normalize(to_tensor(base_img)).to(self.datasource.device)

        # Generates the sticker
        self.adv_sticker = self.craft_sticker(self.mask_circle(base_img))
        save_image(self.datasource.unnormalize(self.adv_sticker), save_path)

    def mask_circle(self, img):
        center = S_SIZE / 2
        rad_2 = math.pow(math.floor(S_SIZE / 2), 2)
        for x in range(S_SIZE):
            for y in range(S_SIZE):
                if math.pow(x - center, 2) + math.pow(y - center, 2) > rad_2:
                    img[0][x][y] = 0
                    img[1][x][y] = 0
                    img[2][x][y] = 0
        return img


    def craft_sticker(self, base_img):
        print("Crafting adversarial sticker...")
        total = 0
        sticker = base_img

        # Iterates for images in the dataset
        for data in self.datasource.get_dataloader(train=False):
            images, labels = data

            # Update the patch for each image if not fooled by the current sticker
            for i, label in enumerate(labels):
                image = images[i]
                transforms = self.get_random_transforms(1)
                results = self.apply_transforms(image[None,:], sticker[None,:], transforms)
                outputs = self.model(results)

                if self.adv_target:
                    # If targeted mode and classified as the target, continue
                    _, predicted = torch.max(outputs.data, 1)
                    if predicted.item() == label:
                        continue
                else:
                    # If untargetted and correct label not in top 5 predictions, continue
                    _, top_predictions = torch.topk(outputs, 5)
                    if label not in top_predictions:
                        continue

                # Otherwise update the sticker for the current image
                sticker = self.update_sticker(image, label, sticker)

            if (total % 10) == 0:
                save_image(self.datasource.unnormalize(sticker), self.save_path)

            total += len(labels)
            if self.limit_samples:
                print("{} de {}".format(total, self.limit_samples))
                if total >= self.limit_samples:
                    break

        return sticker

    def get_random_transforms(self, total):
        transforms = []
        for i in range(total):
            transform = {}
            transform['size_factor'] = random.uniform(0.2, 0.3)
            transform['p_top'] = random.randint(0, I_SIZE - int(S_SIZE * transform['size_factor']))
            transform['p_left'] = random.randint(0, I_SIZE - int(S_SIZE * transform['size_factor']))

            noise_type = random.randint(0, 6)
            if noise_type < 2:
                # Gradient noise
                seed = torch.zeros([1, 1, 1, 2]) if noise_type == 0 else torch.zeros([1, 1, 2, 1])
                seed.uniform_(-self.gradient_noise, self.gradient_noise)
                gradient = interpolate(seed, S_SIZE, mode='bilinear')
                transform['noise'] = self.mask_circle(gradient[0].repeat(3, 1, 1))
            elif noise_type < 5:
                noise = torch.zeros(3, S_SIZE, S_SIZE).uniform_(-self.random_noise, self.random_noise)
                transform['noise'] = self.mask_circle(noise)
            else:
                transform['noise'] = torch.zeros(3, S_SIZE, S_SIZE)

            transform['noise'] = transform['noise'].to(self.datasource.device)
            transforms.append(transform)
        return transforms

    def apply_transforms(self, images, stickers, transforms):
        transformed = torch.zeros_like(images)

        for i, sticker in enumerate(stickers):
            transform = transforms[i]

            # Add noise tensor
            sticker = sticker + transform['noise']

            # Resize
            sticker = interpolate(sticker[None,:], scale_factor=transform['size_factor'], mode='bilinear')[0]

            # Add padding
            padding = (
                transform['p_left'], I_SIZE - transform['p_left'] - sticker.size()[2],
                transform['p_top'], I_SIZE - transform['p_top'] - sticker.size()[1])
            sticker = pad(sticker, padding)

            # If the mask and masked image weren't generated yet
            if 'masked_img' not in transform or 'mask' not in transform:
                transform['mask'] = (sticker != 0).float()
                transform['masked_img'] = (1 - transform['mask']) * images[i]

            # Combines the image with the sticker
            transformed[i] = transform['masked_img'] + transform['mask'] * sticker
        return transformed

    def update_sticker(self, image, label, sticker):
        self.model.eval()
        criterion = CrossEntropyLoss()
        transforms = self.get_random_transforms(self.transforms_per_img)
        images = image.repeat(self.transforms_per_img, 1, 1, 1)
        labels = label.repeat(self.transforms_per_img, 1, 1, 1)

        with torch.enable_grad():
            stickers = sticker.repeat(self.transforms_per_img, 1, 1, 1).requires_grad_(True)
            g_vector = torch.zeros_like(stickers)
            results = self.apply_transforms(images, stickers, transforms)
            outputs = self.model(results)

            for i in range(self.iter_per_img):
                if not self.adv_target:
                    loss = criterion(outputs, labels)
                else:
                    outputs = softmax(outputs, 1)
                    loss = torch.sum(outputs[:,self.adv_target])
                loss.backward()
                stickers_grad = torch.sign(stickers.grad.data)
                stickers.grad.zero_()

                # Update the velocity vector
                flat = torch.flatten(stickers_grad, start_dim=1)
                normalized = flat / torch.reshape(torch.norm(flat, 1, 1), [len(flat), 1])
                g_mean = torch.mean(g_vector, 0)
                g_vector = (g_vector * ((2 / 3) * self.decay_factor) +
                            g_mean * ((1 / 3) * self.decay_factor) +
                            torch.reshape(normalized, stickers_grad.size()))

                # Update the perturbation vector
                stickers = stickers.data + self.alpha * torch.sign(g_vector)
                stickers = self.datasource.clamp(stickers).requires_grad_(True)
                results = self.apply_transforms(images, stickers, transforms)
                outputs = self.model(results)
        return self.mask_circle(torch.mean(stickers, 0).detach())

    def __call__(self, images, labels):
        transforms = self.get_random_transforms(len(labels))
        results = self.apply_transforms(
            images, self.adv_sticker.repeat(len(labels), 1, 1, 1), transforms)
        return results
