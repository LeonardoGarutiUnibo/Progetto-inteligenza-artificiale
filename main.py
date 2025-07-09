from src.train import train_model_simplecnn, train_model_efficientnet
from src.evaluate import evaluate_model
import torch

if __name__ == "__main__":
    print("CUDA disponibile:", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Usando GPU:", torch.cuda.get_device_name(0))
    else:
        print("Usando CPU")

    model, _ = train_model_simplecnn()
    evaluate_model(model = model)
    del model
    torch.cuda.empty_cache()
    model, _ = train_model_efficientnet()
    evaluate_model(model = model)