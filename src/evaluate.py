import torch
from sklearn.metrics import classification_report, confusion_matrix
from .config import DEVICE
from .dataset_utils import get_loaders, load_model

def evaluate_model(model_path="model_best.pt"):
    _, _, test_loader, class_names = get_loaders()

    # ✅ Carica il modello con supporto ai neuroni variabili
    model, neurons_size = load_model(model_path, DEVICE)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    print("✅ Classi caricate da ImageFolder:", class_names)
    print("✅ Valori unici in y_true:", sorted(set(y_true)))
    print("✅ Valori unici in y_pred:", sorted(set(y_pred)))
    # ✅ Class report con mapping esplicito
    print("📊 Report di classificazione:\n")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names
    ))

    # ✅ Confusion matrix (opzionale)
    print("📉 Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))
