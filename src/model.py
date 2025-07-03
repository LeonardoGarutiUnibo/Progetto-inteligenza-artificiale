import torch.nn as nn
import torch.nn.functional as F
from .config import IMAGE_SIZE, NUM_CLASSES


class SimpleCNN(nn.Module):
    def __init__(self, neurons_size=256):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calcolo output dimension dopo 3 pool (diviso 2 tre volte = //8)
        conv_output_size = 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)
        
        self.fc1 = nn.Linear(conv_output_size, neurons_size)
        self.fc2 = nn.Linear(neurons_size, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x