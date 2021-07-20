import urllib
import json
import os

import torch
from PIL import Image
from torchvision import models
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

with open("classes.txt") as file:
    classes = [s.strip() for s in file.readlines()]


def handler(event, context):
    data = urllib.request.urlopen(event["url"])

    image = Image.open(data)
    image = transform(image)
    image = image.unsqueeze(0)

    model = torch.jit.load("./model.pt")
    outputs = model(image)
    target = outputs.argmax().item()

    return {"class": classes[target]}
