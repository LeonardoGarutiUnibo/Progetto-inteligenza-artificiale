from src.train import train_model
from src.evaluate import evaluate_model
import torch

if __name__ == "__main__":

    print("CUDA disponibile:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
    else:
        print("Usando CPU")
    model, _ = train_model()
    evaluate_model()