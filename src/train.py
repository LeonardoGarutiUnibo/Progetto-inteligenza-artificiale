import torch
from torch import nn, optim
from .config import DEVICE, EPOCHS, LR, EARLY_STOP, SEED, NEURONS_SIZE
from .utils import set_seed
from .model import SimpleCNN
from .dataset_utils import get_loaders, save_model
from torch.utils.tensorboard import SummaryWriter
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau



from collections import Counter
import matplotlib.pyplot as plt

from collections import Counter
from .dataset_utils import get_loaders

train_loader, _, _, class_names = get_loaders()
counts = Counter()
for _, labels in train_loader:
    counts.update(labels.tolist())
total = sum(counts.values())
weights = [total / counts[i] for i in range(len(class_names))]
class_weights = torch.FloatTensor(weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

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

def train_model(resume=False):
    train_loader, val_loader, _, class_names = get_loaders()
    
    # Calcola i pesi in base al numero di immagini per classe
    counts = Counter()
    for _, labels in train_loader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(len(class_names))]
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SimpleCNN(neurons_size=NEURONS_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    if resume and os.path.exists("model_best.pt"):
        model.load_state_dict(torch.load("model_best.pt"))
        print("🔄 Checkpoint 'model_best.pt' caricato.")
        
    set_seed(SEED)

    log_dir = os.path.join("runs", f"exp_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    
    no_improve_epochs = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training
        model.train()
        correct_train, total_train = 0, 0
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        cm = confusion_matrix(y_true, y_pred)
        fig_cm = plot_confusion_matrix(cm, class_names)
        writer.add_figure("Confusion_Matrix/Val", fig_cm, epoch)
        plt.close(fig_cm)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
        for i, class_name in enumerate(class_names):
            writer.add_scalar(f"Metrics/{class_name}/Precision", prec[i], epoch)
            writer.add_scalar(f"Metrics/{class_name}/Recall", rec[i], epoch)
            writer.add_scalar(f"Metrics/{class_name}/F1", f1[i], epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, neurons_size=NEURONS_SIZE, path="model_best.pt")
            print("Miglior modello salvato in model_best.pt")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"Nessun miglioramento in {no_improve_epochs} epoche")

        if no_improve_epochs >= EARLY_STOP:
            print(f"Early stopping attivato dopo {no_improve_epochs} epoche senza miglioramento.")
            break

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                writer.add_histogram(f"Weights/{name}", param.data.cpu(), epoch)
                writer.add_histogram(f"Gradients/{name}", param.grad.cpu(), epoch)

    writer.close()
    save_model(model, neurons_size=NEURONS_SIZE, path="model_final.pt")
    print("Modello finale salvato in model_final.pt")
    return model, class_names