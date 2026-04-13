"""
Fashion Recommendation Model Trainer (Memory-Optimised)
=========================================================
Requirements: pip install tensorflow numpy pandas opencv-python

Run: python train_model.py
Output: fashion-recommendation1.h5
"""

import numpy as np
import pandas as pd
import cv2
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Concatenate, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 64          # reduced from 128 → 4x less memory
EPOCHS       = 20
BATCH_SIZE   = 8           # small batch to save RAM
CSV_PATH     = "ratings_data.csv"
TOPWEAR_DIR  = "fcs_dataset/topwear"
BOTTOM_DIR   = "fcs_dataset/bottomwear"
FOOT_DIR     = "fcs_dataset/footwear"
MODEL_OUT    = "fashion-recommendation1.h5"
# ──────────────────────────────────────────────────────────────────────────────


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32") / 255.0


def parse_combination(combo_str):
    parts = []
    i = 0
    while i < len(combo_str):
        end = combo_str.find(".jpg", i) + 4
        parts.append(combo_str[i:end])
        i = end
    return parts


def build_dataset(df):
    X1, X2, X3 = [], [], []
    for _, row in df.iterrows():
        parts = parse_combination(row["Combination"])
        X1.append(load_image(os.path.join(TOPWEAR_DIR, parts[0])))
        X2.append(load_image(os.path.join(BOTTOM_DIR,  parts[1])))
        X3.append(load_image(os.path.join(FOOT_DIR,    parts[2])))
    return (np.array(X1, dtype="float32"),
            np.array(X2, dtype="float32"),
            np.array(X3, dtype="float32"))


def cnn_block(inp, filters=16, name_prefix=""):
    """Lightweight CNN + GlobalAveragePooling → tiny fixed output vector."""
    x = Conv2D(filters, (3, 3), activation="relu", padding="same",
               name=f"{name_prefix}_conv1")(inp)
    x = MaxPooling2D()(x)
    x = Conv2D(filters * 2, (3, 3), activation="relu", padding="same",
               name=f"{name_prefix}_conv2")(x)
    x = GlobalAveragePooling2D()(x)   # outputs (filters*2,) — much smaller than Flatten
    return x


def build_model():
    input1  = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="topwear_input")
    input2  = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="bottomwear_input")
    input3  = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="footwear_input")
    occ_inp = Input(shape=(3,),                    name="occasion_input")

    x1 = cnn_block(input1, filters=16, name_prefix="top")   # → 32-d
    x2 = cnn_block(input2, filters=16, name_prefix="bot")   # → 32-d
    x3 = cnn_block(input3, filters=16, name_prefix="foot")  # → 32-d

    combined = Concatenate()([x1, x2, x3, occ_inp])         # → 99-d

    x = Dense(64, activation="relu")(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid", name="rating_output")(x)

    model = Model(inputs=[input1, input2, input3, occ_inp], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset ...")
    df = pd.read_csv(CSV_PATH)
    total = len(df)

    # Cap at 500 rows to avoid RAM crash
    if total > 500:
        df = df.sample(500, random_state=42).reset_index(drop=True)
        print(f"Sampled 500 rows from {total} to save memory")

    print(f"Rows: {len(df)}")

    print("Loading images ...")
    X1, X2, X3 = build_dataset(df)
    print(f"Image arrays shape: {X1.shape}")

    occasion = to_categorical(df["Occasion"] - 1, num_classes=3).astype("float32")
    rating   = df["Rating"].values.astype("float32") / 10.0

    print("Building model ...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
    ]

    print("Training ...")
    model.fit(
        [X1, X2, X3, occasion],
        rating,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
    )

    model.save(MODEL_OUT)
    print(f"\nModel saved -> {MODEL_OUT}")
