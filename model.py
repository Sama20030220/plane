import torch.nn as nn
import torchvision.models as models


def build_model(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
