import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
import csv
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.model import build_cnn_lstm_model, convert_to_tflite
from configs.config import (
    DATA_DIR, CLASSES, IMG_HEIGHT, IMG_WIDTH,
    SEQUENCE_LENGTH, MODEL_PATH, TFLITE_MODEL_PATH, LOGS_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, FINE_TUNE_LAYERS,
    USE_MIXED_PRECISION,
)

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision (float16) ativado")


# ---------------------------------------------------------------------------
# Utilidades de logging
# ---------------------------------------------------------------------------

def make_run_dir() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(LOGS_DIR, f"run-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_history(history, run_dir: str):
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(run_dir, "history.csv")
    keys = list(history.history.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        epochs = len(history.history[keys[0]]) if keys else 0
        for e in range(epochs):
            writer.writerow([e + 1] + [history.history[k][e] for k in keys])

    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        if "loss" in history.history:
            ax[0].plot(history.history["loss"], label="train_loss")
        if "val_loss" in history.history:
            ax[0].plot(history.history["val_loss"], label="val_loss")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        if "accuracy" in history.history:
            ax[1].plot(history.history["accuracy"], label="train_acc")
        if "val_accuracy" in history.history:
            ax[1].plot(history.history["val_accuracy"], label="val_acc")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "curves.png"), dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Aviso: falha ao salvar curvas: {e}")


def save_eval_artifacts(model, X_test, y_test, run_dir: str):
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    y_pred = (y_prob > 0.5).astype(np.int32)

    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4, zero_division=0)
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    with open(os.path.join(run_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, ensure_ascii=False, indent=2)

    try:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Aviso: falha ao salvar matriz de confusão: {e}")

    try:
        np.save(os.path.join(run_dir, "y_test.npy"), y_test)
        np.save(os.path.join(run_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(run_dir, "y_prob.npy"), y_prob)
    except Exception as e:
        print(f"Aviso: falha ao salvar arrays: {e}")


def save_run_metadata(run_dir: str, extra: dict):
    meta = {
        "timestamp": os.path.basename(run_dir).replace("run-", ""),
        "data_dir": str(DATA_DIR),
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "sequence_length": SEQUENCE_LENGTH,
        "classes": CLASSES,
        "model_path": str(MODEL_PATH),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "fine_tune_layers": FINE_TUNE_LAYERS,
        "mixed_precision": USE_MIXED_PRECISION,
    }
    meta.update(extra or {})
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Carregamento de dados com tf.data (memory-efficient)
# ---------------------------------------------------------------------------

def extract_sequences_from_video(video_path: str, label: int):
    """Extrai sequências de frames de um vídeo usando sliding window."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    sequences = []
    if len(frames) >= SEQUENCE_LENGTH:
        step = max(1, SEQUENCE_LENGTH // 2)
        for start in range(0, len(frames) - SEQUENCE_LENGTH + 1, step):
            sequences.append(np.array(frames[start:start + SEQUENCE_LENGTH]))

    return sequences, label, os.path.basename(video_path), len(frames)


def build_dataset():
    """
    Carrega vídeos e retorna arrays numpy de sequências.
    Usa float32 com resolução configurável para reduzir uso de RAM.
    """
    all_sequences = []
    all_labels = []

    print("=" * 50)
    print("CARREGANDO DATASET")
    print("=" * 50)

    for class_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"AVISO: Pasta '{class_dir}' não encontrada!")
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(('.avi', '.mp4'))]
        print(f"\nClasse '{class_name}': {len(files)} vídeos encontrados")

        if not files:
            print(f"   ERRO: Nenhum vídeo na pasta {class_name}!")
            continue

        for file_name in files:
            video_path = os.path.join(class_dir, file_name)
            sequences, label, name, n_frames = extract_sequences_from_video(
                video_path, class_index
            )
            if sequences:
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
                print(f"   {name}: {n_frames} frames -> {len(sequences)} amostras")
            else:
                print(f"   {name}: {n_frames} frames (insuficiente, mínimo: {SEQUENCE_LENGTH})")

    if not all_sequences:
        return np.array([]), np.array([])

    return np.array(all_sequences, dtype=np.float32), np.array(all_labels)


def create_augmented_dataset(X, y, batch_size):
    """
    Cria tf.data.Dataset com data augmentation em tempo real.
    Augmentations aplicadas aleatoriamente a cada sequência:
    - flip horizontal (espelha a cena)
    - variação de brilho
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def augment(sequence, label):
        if tf.random.uniform([]) > 0.5:
            sequence = tf.image.flip_left_right(
                tf.reshape(sequence, [-1, IMG_HEIGHT, IMG_WIDTH, 3])
            )
            sequence = tf.reshape(sequence, [SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3])

        if tf.random.uniform([]) > 0.5:
            delta = tf.random.uniform([], -0.1, 0.1)
            sequence = tf.clip_by_value(sequence + delta, 0.0, 1.0)

        return sequence, label

    dataset = (
        dataset
        .shuffle(len(X), reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = build_dataset()

    run_dir = make_run_dir()
    save_run_metadata(
        run_dir,
        extra={
            "n_samples": int(len(X)),
            "class_distribution": (
                {CLASSES[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
                if len(y) > 0 else {}
            ),
        },
    )

    print("\n" + "=" * 50)
    print("RESUMO DO DATASET")
    print("=" * 50)
    print(f"Total de amostras: {len(X)}")
    ram_mb = X.nbytes / (1024 * 1024) if len(X) > 0 else 0
    print(f"Uso de RAM estimado (dados): {ram_mb:.0f} MB")

    if len(X) == 0:
        print("\nERRO CRÍTICO: Nenhum dado encontrado!")
        print("Verifique se você:")
        print("  1. Baixou os vídeos ADL (adl-*-cam0-rgb.zip) do UR Fall")
        print("  2. Rodou 'python scripts/prepare_ur_fall.py'")
        exit(1)

    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  - {CLASSES[idx]}: {count} amostras")

    if len(unique) < 2:
        print("\nERRO CRÍTICO: Apenas uma classe encontrada!")
        print("O modelo precisa de exemplos de AMBAS as classes (Normal e Fall).")
        print("Baixe os vídeos ADL do UR Fall Dataset.")
        exit(1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTreino: {len(X_train)} amostras")
    print(f"Teste:  {len(X_test)} amostras")

    # Dataset com augmentation
    train_ds = create_augmented_dataset(X_train, y_train, BATCH_SIZE)
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Modelo
    print("\n" + "=" * 50)
    print("CONSTRUINDO MODELO")
    print("=" * 50)
    model = build_cnn_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        fine_tune_layers=FINE_TUNE_LAYERS,
        learning_rate=LEARNING_RATE,
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True,
            monitor='val_accuracy', mode='max', verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, "keras_history.csv"), append=False,
        ),
    ]

    # Treinar
    print("\n" + "=" * 50)
    print("INICIANDO TREINAMENTO")
    print("=" * 50)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    save_history(history, run_dir)

    # Avaliação
    print("\n" + "=" * 50)
    print("AVALIAÇÃO FINAL")
    print("=" * 50)
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")
    print(f"Loss no conjunto de teste: {loss:.4f}")
    print(f"\nModelo Keras salvo em: {MODEL_PATH}")

    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"loss": float(loss), "accuracy": float(accuracy)}, f, ensure_ascii=False, indent=2)

    save_eval_artifacts(model, X_test, y_test, run_dir)

    # Converter para TFLite
    print("\n" + "=" * 50)
    print("CONVERTENDO PARA TFLITE")
    print("=" * 50)
    try:
        convert_to_tflite(model, str(TFLITE_MODEL_PATH), quantize=True)
    except Exception as e:
        print(f"Aviso: falha na conversão TFLite: {e}")

    print(f"\nLogs salvos em: {run_dir}")
