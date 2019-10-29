import torch.nn as nn
import torch.optim as optim
from src.base import Task, Defense
from src.utils.functions import save_weights


class Train(Task):
    '''
    Task to train a model for the given datasource.

    Code based in PyTorch's tutorial 'Training a Classifier'.

    If `attack` is given, the model will be adversarialy trained, applying the attack
    to the samples before using them.
    If an `src.base.defense.Defense` instance is passed as net, the defense must be.
    differentiable in order to work.

    Args:
        epochs (int, optional): number of epochs to execute.
        lr (float, optional): used learning rate.
        momentum (float, optional): used momentum.
        weight_decay (float, optional): used weight decay.
    '''
    def __init__(self, epochs=1, lr=0.001, momentum=0.9, weight_decay=0.0001, **kwargs):
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def exec_task_simple(self, results_path, net, datasource, attack=None):
        result = {}
        criterion = nn.CrossEntropyLoss()
        percentage_factor = 100/datasource.get_dataset(True).__len__()

        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)

        net.train()
        for epoch in range(self.epochs):
            total = 0
            print('Epoch: {}/{} - {:.2f} %'.format(
                epoch + 1, self.epochs, total*percentage_factor), end='\r')

            for i, data in enumerate(datasource.get_dataloader(train=True), 0):
                inputs, labels = data
                optimizer.zero_grad()
                total += labels.size()[0]

                # Apply the attack if given
                if attack:
                    inputs = attack(inputs, labels)

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                print('Epoch: {}/{} - {:.2f} %'.format(
                    epoch + 1, self.epochs, total*percentage_factor), end='\r')

        # Uses the data added in Scheduler to get the net_id and save the weights
        save_weights(net, net.data['net_id'])
        return result
