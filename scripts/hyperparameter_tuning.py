"""
Script industrial pentru optimizarea automată a hiperparametrilor folosind Keras-Tuner.

Acest script realizează:
1. Definirea unui spațiu de căutare dinamic (straturi Dense, unități și rate de Dropout).
2. Configurarea algoritmului de căutare RandomSearch.
3. Executarea ciclului de optimizare pe datasetul local de fructe.
4. Extragerea și afișarea celei mai bune configurații arhitecturale găsite.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch

# Maparea directoarelor de date
DATA_DIR = os.path.join(
    os.path.expanduser("~"), "Desktop", "Cap. 10 Keras", "dataset_fructe"
)

# Configurații de bază pentru imagini
IMG_SIZE = (224, 224)
BATCH_SIZE = 2
NUM_CLASSES = 3

# Generator minimalist de date (fără augmentare agresivă pentru a accelera căutarea)
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_generator = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)


def build_tunable_model(hp):
    """
    Construiește o arhitectură Keras cu hiperparametri variabili controlați de Keras-Tuner.
    """
    # Încărcăm modelul de bază înghețat
    base_model = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 1. OPTIMIZARE UNITĂȚI: Lăsăm tunerul să aleagă între 32 și 256 de neuroni
    tunable_units = hp.Int("dense_units", min_value=32, max_value=256, step=32)
    x = Dense(units=tunable_units, activation="relu")(x)

    # 2. OPTIMIZARE REGULARIZARE: Tunerul va testa rate de Dropout între 0.1 și 0.5
    tunable_dropout = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    x = Dropout(rate=tunable_dropout)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # 3. OPTIMIZARE LEARNING RATE: Tunerul va alege rata optimă de învățare pentru Adam
    tunable_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=tunable_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


print("[INFO] Configurare motor de căutare Keras-Tuner (RandomSearch)...")
tuner = RandomSearch(
    build_tunable_model,
    objective="val_accuracy",
    max_trials=3,  # Câte configurații diferite va testa în total
    executions_per_trial=1,  # Câte rulări per configurație pentru stabilizare
    directory="keras_tuner_dir",
    project_name="agrivision_tuning",
    overwrite=True,
)

print("[INFO] Pornire rulări experimentale automate...")
tuner.search(train_generator, validation_data=val_generator, epochs=1)

print("\n" + "=" * 40)
print("[SUCCESS] Căutarea automată s-a finalizat!")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"-> Numărul optim de neuroni în stratul Dense: {best_hps.get('dense_units')}")
print(f"-> Rata optimă de Dropout pentru regularizare: {best_hps.get('dropout_rate')}")
print(f"-> Rata optimă de învățare (Learning Rate): {best_hps.get('learning_rate')}")
print("=" * 40)
