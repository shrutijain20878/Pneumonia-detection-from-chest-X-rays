import torch
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_resnet(model_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model


def load_inception(model_path):
    model = models.inception_v3(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model
