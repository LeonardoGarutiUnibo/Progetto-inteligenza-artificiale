import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Numero di versioni aumentate da creare per ciascuna immagine
AUGMENTATIONS_PER_IMAGE = 3

# Trasformazioni
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

def augment_image(image, n):
    """Restituisce una lista di n immagini aumentate."""
    return [augmentation_transforms(image) for _ in range(n)]

def augment_in_place(input_dir, augmentations_per_image=AUGMENTATIONS_PER_IMAGE):
    """Effettua data augmentation direttamente nella stessa cartella."""
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in tqdm(os.listdir(class_path), desc=f"Augmenting '{class_name}'"):
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path).convert("RGB") as img:
                    augmented_imgs = augment_image(img, augmentations_per_image)
                    base_name = os.path.splitext(img_name)[0]

                    for i, aug_img in enumerate(augmented_imgs):
                        save_name = f"{base_name}_aug{i}.jpg"
                        save_path = os.path.join(class_path, save_name)
                        aug_img.save(save_path)

            except Exception as e:
                print(f"Errore con {img_path}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data augmentation in-place (nella stessa cartella).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory con le classi (es: dataset/train)")
    parser.add_argument("--n", type=int, default=AUGMENTATIONS_PER_IMAGE, help="Numero di immagini aumentate per ciascuna immagine")
    args = parser.parse_args()

    augment_in_place(args.input_dir, args.n)