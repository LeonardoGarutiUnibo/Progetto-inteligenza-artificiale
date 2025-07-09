import torch
from torch import nn, optim
from .config import DEVICE, EPOCHS, LR, EARLY_STOP, SEED, NEURONS_SIZE, BATCH_SIZE, AVG_POOL, KERNEL_SIZE, STRIDE, PADDING
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
from .train_utils import compute_class_weights, get_efficientnet, plot_confusion_matrix, count_images_per_class, plot_class_distribution



from collections import Counter
import matplotlib.pyplot as plt

from collections import Counter
from .dataset_utils import get_loaders

def log_hparams(writer, hparams, metrics):
    with SummaryWriter(os.path.join(writer.log_dir, "hparam_tuning")) as hp_writer:
        hp_writer.add_hparams(hparams, metrics)

def compute_class_weights_from_loader(train_loader, class_names):
    counts = Counter()
    for _, labels in train_loader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(len(class_names))]
    return torch.FloatTensor(weights).to(DEVICE)


def setup_tensorboard(log_prefix):
    log_dir = os.path.join("runs", f"{log_prefix}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    writer.add_hparams(
        {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "early_stop": EARLY_STOP,
            "neurons": NEURONS_SIZE,
            "avg_pool": AVG_POOL,
            "kernel_size": KERNEL_SIZE,
            "stride": STRIDE,
            "padding": PADDING
        },
        {
            "hparam/accuracy": 0,
            "hparam/loss": 0
        }
    )
    return writer


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, class_names, checkpoint_path):
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        correct_train, total_train, total_loss = 0, 0, 0
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

        model.eval()
        val_loss, correct, total = 0, 0, 0
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

        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('losses',{"Loss/Train": train_loss, "Loss/Val": val_loss}, epoch)
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
            save_model(model, neurons_size=NEURONS_SIZE, path=checkpoint_path)
            print(f"✅ Salvataggio modello migliore in {checkpoint_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= EARLY_STOP:
            print("⏹️ Early stopping")
            break
    metrics = {
        "hparam/accuracy": val_acc,
        "hparam/loss": val_loss
    }
    hparams = {
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stop": EARLY_STOP,
        "neurons": NEURONS_SIZE,
        "avg_pool": AVG_POOL,
        "kernel_size": KERNEL_SIZE,
        "stride": STRIDE,
        "padding": PADDING
    }
    log_hparams(writer, hparams, metrics)
    return model


def train_model_simplecnn(resume=False):
    set_seed(SEED)
    train_loader, val_loader, _, class_names = get_loaders()
    class_weights = compute_class_weights_from_loader(train_loader, class_names)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = SimpleCNN(neurons_size=NEURONS_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    if resume and os.path.exists("model_best.pt"):
        model.load_state_dict(torch.load("model_best.pt"))
        print("🔄 Checkpoint 'model_best.pt' caricato.")

    writer = setup_tensorboard("simplecnn")

    sample_input = next(iter(train_loader))[0][:1].to(DEVICE)  # un batch con una sola immagine
    writer.add_graph(model, sample_input)

    model = run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, class_names, "model_best.pt")
    save_model(model, neurons_size=NEURONS_SIZE, path="model_final_simplecnn.pt")
    
    writer.close()
    return model, class_names


def train_model_efficientnet(resume=False):
    torch.cuda.empty_cache()
    set_seed(SEED)
    train_loader, val_loader, _, class_names = get_loaders()
    class_weights = compute_class_weights(train_loader, len(class_names), DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = get_efficientnet(num_classes=len(class_names), neurons_size=NEURONS_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    if resume and os.path.exists("model_efficientnet.pt"):
        model.load_state_dict(torch.load("model_efficientnet.pt"))
        print("🔄 Checkpoint caricato.")

    writer = setup_tensorboard("effnet")

    sample_input = next(iter(train_loader))[0][:1].to(DEVICE)
    writer.add_graph(model, sample_input)

    model = run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, class_names, "model_best_efficientnet.pt")
    save_model(model, neurons_size=NEURONS_SIZE, path="model_final_efficientnet.pt")
    writer.close()
    return model, class_names
