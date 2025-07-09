import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_masks(with_bg_dir, no_bg_dir, out_mask_dir):
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)
    files = os.listdir(with_bg_dir)

    for f in tqdm(files, desc="Generazione maschere"):
        img_with_bg = cv2.imread(os.path.join(with_bg_dir, f))
        img_no_bg = cv2.imread(os.path.join(no_bg_dir, f).replace(".jpeg", ".png"), cv2.IMREAD_UNCHANGED)  # BGRA

        if img_with_bg is None or img_no_bg is None:
            print(f"Immagine mancante o danneggiata: {f}")
            continue

        if img_no_bg.shape[2] == 4:
            alpha = img_no_bg[:, :, 3]
            mask = (alpha > 0).astype(np.uint8) * 255
        else:
            diff = cv2.absdiff(img_with_bg, img_no_bg[:, :, :3])
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mask = (gray > 10).astype(np.uint8) * 255

        out_path = os.path.join(out_mask_dir, os.path.splitext(f)[0] + ".png")
        cv2.imwrite(out_path, mask)

if __name__ == "__main__":
    classes = ["adventure", "naked", "offroad", "sportive"]
    folders = ["training", "validation", "test"]
    for i in range(len(classes)):
        for y in range (len(folders)):
            generate_masks(
                with_bg_dir="data/"+folders[y]+"/"+classes[i],
                no_bg_dir="data_noBackground/"+folders[y]+"/"+classes[i],
                out_mask_dir="mask/"+folders[y]+"/"+classes[i]
            )
    