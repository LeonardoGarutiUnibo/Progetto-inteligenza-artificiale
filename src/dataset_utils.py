from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, NEURONS_SIZE
import torch
from .model import SimpleCNN

#  Trasformazioni con augmentazione per il TRAINING
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomAffine(15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Trasformazioni standard per VALIDATION e TEST
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_loaders():
    train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_set = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_set = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_set.classes


def save_model(model, neurons_size, path):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'neurons_size': neurons_size
    }
    torch.save(save_dict, path)

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    neurons_size = checkpoint.get('neurons_size', NEURONS_SIZE)
    model = SimpleCNN(neurons_size=neurons_size).to(device)

    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']

    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    return model, neurons_size