import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import itertools
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import Counter
import torch
from torchvision import models
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .config import NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE, NEURONS_SIZE

def compute_class_weights(loader, num_classes, device):
    counts = Counter()
    for _, labels in loader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(num_classes)]
    return torch.FloatTensor(weights).to(device)

def compute_class_weights(loader, num_classes, device):
    counts = Counter()
    for _, labels in loader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    weights = [total / counts[i] if counts[i] > 0 else 0 for i in range(num_classes)]
    return torch.FloatTensor(weights).to(device)


def get_efficientnet(model_name="efficientnet_b0", num_classes=NUM_CLASSES, neurons_size=NEURONS_SIZE, pretrained=True):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} non supportato.")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, neurons_size),
        nn.ReLU(),
        nn.Linear(neurons_size, num_classes)
    )

    return model

def count_images_per_class(loader, class_names):
    counts = Counter()
    for _, labels in loader:
        counts.update(labels.tolist())
    counts_list = [counts[i] for i in range(len(class_names))]
    return counts_list

def plot_class_distribution(counts, class_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(class_names, counts)
    ax.set_xlabel("Classi")
    ax.set_ylabel("Numero immagini")
    ax.set_title("Distribuzione immagini per classe")
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label'
    )
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def apply_gradcam(model, image_tensor, target_class=None, target_layer=None):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)
    
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Trova il layer da monitorare
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = target_class if target_class is not None else output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_val[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))
    cam -= np.min(cam)
    cam /= np.max(cam)

    handle_fw.remove()
    handle_bw.remove()

    return cam, class_idx

def show_gradcam_on_image(img_tensor, cam, alpha=0.5):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = img - img.min()
    img = img / img.max()
    img = np.uint8(255 * img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = np.uint8(heatmap * alpha + img * (1 - alpha))

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

