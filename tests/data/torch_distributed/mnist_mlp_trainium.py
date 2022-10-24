import os
import time
import torch
from model import MLP

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
import torch_xla.core.xla_model as xm

# XLA imports for parallel loader and multi-processing
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data.distributed import DistributedSampler

# Initialize XLA process group for torchrun
import torch_xla.distributed.xla_backend

torch.distributed.init_process_group("xla")

# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 32

# Load MNIST train dataset
train_dataset = mnist.MNIST(
    root=os.path.join("./MNIST_DATA_train", str(xm.get_ordinal())),
    train=True,
    download=True,
    transform=ToTensor(),
)


def main():
    # XLA MP: get world size
    world_size = xm.xrt_world_size()
    # multi-processing: ensure each worker has same initial weights
    torch.manual_seed(0)

    # Move model to device and declare optimizer and loss function
    device = "xla"
    model = MLP().to(device)
    # For multiprocessing, scale up learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01 * world_size)
    loss_fn = torch.nn.NLLLoss()

    # Prepare data loader
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=xm.get_ordinal(), shuffle=True
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
    )
    # XLA MP: use MpDeviceLoader from torch_xla.distributed
    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    # Run the training loop
    print("----------Training ---------------")
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
        print(f"Epoch: {epoch}")
        for idx, (train_x, train_label) in enumerate(train_device_loader):
            optimizer.zero_grad()
            train_x = train_x.view(train_x.size(0), -1)
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            xm.optimizer_step(optimizer)  # XLA MP: performs grad allreduce and optimizer step
            if idx < WARMUP_STEPS:  # skip warmup iterations
                start = time.time()

    # Compute statistics for the last epoch
    interval = idx - WARMUP_STEPS  # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    print("Final loss is {:0.4f}".format(loss.detach().to("cpu")))

    # Save checkpoint for evaluation (xm.save ensures only one process save)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, "checkpoints/checkpoint.pt")

    print("----------End Training ---------------")


if __name__ == "__main__":
    main()
