import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.autograd import Variable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Net(nn.Module):
    # Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self):
        logger.info("Create neural network module")

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _load_hyperparameters(hyperparameters):
    logger.info("Load hyperparameters")
    # backend for distributed training (default: None')
    backend = hyperparameters.get('backend', None)
    # batch size for training (default: 64)
    batch_size = hyperparameters.get('batch_size', 60)
    # batch size for testing (default: 1000)
    test_batch_size = hyperparameters.get('test_batch_size', 1000)
    # number of epochs to train (default: 10)
    epochs = hyperparameters.get('epochs', 3)
    # learning rate (default: 0.01)
    lr = hyperparameters.get('lr', 0.01)
    # SGD momentum (default: 0.5)
    momentum = hyperparameters.get('momentum', 0.5)
    # random seed (default: 1)
    seed = hyperparameters.get('seed', 1)
    # how many batches to wait before logging training status
    log_interval = hyperparameters.get('log_interval', 100)
    logger.info(
        'backend: {}, batch_size: {}, test_batch_size: {}, '.format(backend, batch_size, test_batch_size) +
        'epochs: {}, lr: {}, momentum: {}, seed: {}, log_interval: {}'.format(epochs, lr, momentum, seed, log_interval)
    )
    return backend, batch_size, test_batch_size, epochs, lr, momentum, seed, log_interval


def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(training_dir, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def train(channel_input_dirs, num_gpus, hosts, host_rank, master_addr, master_port, hyperparameters):
    training_dir = channel_input_dirs['training']
    backend, batch_size, test_batch_size, epochs, lr, momentum, \
    seed, log_interval = _load_hyperparameters(hyperparameters)
    is_distributed = hosts > 1 and backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    cuda = num_gpus > 0
    logger.debug("Number of gpus available - {}".format(num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(backend=backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            backend, dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
            dist.get_rank(), torch.cuda.is_available(), num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    train_loader = _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(test_batch_size, training_dir, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = Net()
    if cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed:
                _average_gradients(model)
            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                           100. * batch_idx / len(train_loader), loss.data[0]))
        test(model, test_loader, cuda)
    return model


def test(model, test_loader, cuda):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    logger.debug('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
