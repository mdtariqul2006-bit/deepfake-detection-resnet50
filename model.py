import torch
import torchvision
import torch.nn as nn


def imageModel(num_classes: int = 2, seed: int = 42):
    torch.manual_seed(seed)

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    preprocessing_transforms = weights.transforms()
    model = torchvision.models.resnet50(weights=weights)

    # Freeze early backbone layers, unfreeze layer4 + fc for fine-tuning
    for name, param in model.named_parameters():
        if "layer 3" or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )

    return model, preprocessing_transforms