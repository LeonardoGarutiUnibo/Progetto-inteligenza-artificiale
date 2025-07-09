import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from .config import DEVICE
from .dataset_utils import get_loaders, load_model
import time


def evaluate_model(model=None, model_path="model_best.pt"):
    _, _, test_loader, class_names = get_loaders()
    num_classes = len(class_names)

    if model is None:
        model, _ = load_model(model_path, DEVICE)

    model.eval()

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, preds = torch.max(probs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_scores.extend(probs.cpu().numpy())

    print("Classi caricate da ImageFolder:", class_names)
    print("Valori unici in y_true:", sorted(set(y_true)))
    print("Valori unici in y_pred:", sorted(set(y_pred)))
    print("📊 Report di classificazione:\n")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names
    ))
    print("📉 Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_scores = np.array(y_scores)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    writer = SummaryWriter(log_dir="runs/eval_" + time.strftime("%Y%m%d-%H%M%S"))

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"Classe {class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC Multiclasse")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    writer.add_figure("ROC_Curve", plt.gcf())
    writer.close()
    plt.close()