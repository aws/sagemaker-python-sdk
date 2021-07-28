import torch
from torchvision.models import resnet

model = resnet.resnet34(pretrained=True)
model.eval()

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
traced_model.save("model.pt")
