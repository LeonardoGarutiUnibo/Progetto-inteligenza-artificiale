import torch.nn as nn
import torch.nn.functional as F
import torch
from .config import IMAGE_SIZE, NUM_CLASSES, NEURONS_SIZE, AVG_POOL, KERNEL_SIZE, STRIDE, PADDING


class SimpleCNN(nn.Module):
    def __init__(self, neurons_size=NEURONS_SIZE):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        if AVG_POOL:
            self.pool = nn.AvgPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            conv_output_size = 128
        else:
            self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)   
            conv_output_size = 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)

        print("IMAGE_SIZE:", IMAGE_SIZE)
        print("Computed conv_output_size:", conv_output_size)
        self.fc1 = nn.Linear(conv_output_size, neurons_size)
        self.fc2 = nn.Linear(neurons_size, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        if AVG_POOL:
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = torch.flatten(x, 1)
        else:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
