import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

ROOT_DIR = 'data_noBackground'
VALID_EXTENSIONS = ('.jpg', '.jpeg',)

def remove_background_from_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        output = remove(img)
        return output

def process_all_images(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Processing in {subdir}"):
            if file.lower().endswith(VALID_EXTENSIONS):
                file_path = os.path.join(subdir, file)
                try:
                    no_bg = remove_background_from_image(file_path)

                    new_file_path = os.path.splitext(file_path)[0] + ".png"

                    no_bg.save(new_file_path)

                    if not file.lower().endswith('.png'):
                        os.remove(file_path)

                except Exception as e:
                    print(f"❌ Errore con {file_path}: {e}")

if __name__ == "__main__":
    process_all_images(ROOT_DIR)
    print("✅ Rimozione sfondi completata e immagini originali rimosse.")