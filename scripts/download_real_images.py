"""Script industrial pentru generarea unui volum crescut de date geometrice.

Generează 40 imagini/clasă și introduce variabilitate structurală masivă
pentru a preveni colapsul modului.
"""

import os
import random

from PIL import Image, ImageDraw
import numpy as np

# Configurare căi dinamice bazate pe noua arhitectură a proiectului
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dataset_fructe")

structura_foldere = [
    "train/mere",
    "train/banane",
    "train/portocale",
    "val/mere",
    "val/banane",
    "val/portocale",
]

print("[INFO] Curatare si generare dataset extins (40 imagini per clasa)...")

for subpath in structura_foldere:
    folder_complet = os.path.join(DATA_DIR, subpath.replace("/", os.sep))
    os.makedirs(folder_complet, exist_ok=True)

    # Curățăm folderul complet de orice test vechi
    for file in os.listdir(folder_complet):
        os.remove(os.path.join(folder_complet, file))

    # Generăm 40 de imagini unice pentru antrenare și 10 pentru validare
    limita_imagini = 40 if "train" in subpath else 10

    for i in range(1, limita_imagini + 1):
        intensitate = random.randint(170, 230)
        matrice = np.ones((224, 224, 3), dtype=np.uint8) * intensitate
        img = Image.fromarray(matrice)
        draw = ImageDraw.Draw(img)

        offset_x = random.randint(-35, 35)
        offset_y = random.randint(-35, 35)
        raza_mod = random.randint(-20, 20)

        if "mere" in subpath:
            # Cerc roșu (mere)
            draw.ellipse(
                [
                    35 + offset_x,
                    35 + offset_y,
                    185 + offset_x + raza_mod,
                    185 + offset_y + raza_mod,
                ],
                fill=(240, 20, 20),
            )
        elif "banane" in subpath:
            # Triunghi galben (banane)
            draw.polygon(
                [(35 + offset_x, 190), (112, 35 + offset_y), (190 + offset_x, 190)],
                fill=(245, 245, 20),
            )
        elif "portocale" in subpath:
            # Cerc portocaliu (portocale)
            draw.ellipse(
                [
                    35 + offset_x,
                    35 + offset_y,
                    185 + offset_x + raza_mod,
                    185 + offset_y + raza_mod,
                ],
                fill=(250, 140, 10),
            )

        # Adăugăm texturi aleatorii de linii negre pentru margini convoluționale complexe
        for _ in range(random.randint(1, 3)):
            draw.line(
                [
                    (random.randint(0, 40), random.randint(0, 224)),
                    (random.randint(180, 224), random.randint(0, 224)),
                ],
                fill=(0, 0, 0),
                width=random.randint(1, 2),
            )

        img.save(os.path.join(folder_complet, f"img_real_{i}.jpg"), "JPEG")

print("[SUCCESS] Dataset extins generat cu succes pe disc!")
