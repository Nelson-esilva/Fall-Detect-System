import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
    Carrega vídeos e retorna arrays numpy de sequências, labels e IDs de vídeo.
    O video_id permite fazer split por vídeo (evita data leakage).
    """
    all_sequences = []
    all_labels = []
    all_video_ids = []
    video_counter = 0

    print("=" * 50)
    print("CARREGANDO DATASET")
    print("=" * 50)

    for class_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"AVISO: Pasta '{class_dir}' não encontrada!")
            continue

        files = sorted(f for f in os.listdir(class_dir) if f.endswith(('.avi', '.mp4')))
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
                all_video_ids.extend([video_counter] * len(sequences))
                print(f"   {name}: {n_frames} frames -> {len(sequences)} amostras (video_id={video_counter})")
            else:
                print(f"   {name}: {n_frames} frames (insuficiente, mínimo: {SEQUENCE_LENGTH})")
            video_counter += 1

    if not all_sequences:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(all_sequences, dtype=np.float32),
        np.array(all_labels),
        np.array(all_video_ids),
    )


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
    import argparse

    parser = argparse.ArgumentParser(description="Treinamento do modelo de detecção de quedas")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                        choices=["cnn_lstm", "cnn_only", "cnn_lstm_frozen"],
                        help="Arquitetura do modelo (default: cnn_lstm)")
    parser.add_argument("--fine-tune-layers", type=int, default=None,
                        help="Override de camadas a descongelar (default: config.py)")
    parser.add_argument("--lstm-units", type=int, default=64,
                        help="Unidades da LSTM (default: 64)")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Desativar data augmentation")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override da resolução (ex: 224)")
    parser.add_argument("--split", type=str, default="video",
                        choices=["video", "window"],
                        help="Estratégia de split: 'video' (sem leakage) ou 'window' (legacy)")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag descritiva para o run (salva nos metadados)")
    args = parser.parse_args()

    img_h = args.img_size or IMG_HEIGHT
    img_w = args.img_size or IMG_WIDTH
    ft_layers = args.fine_tune_layers if args.fine_tune_layers is not None else FINE_TUNE_LAYERS

    X, y, video_ids = build_dataset()

    run_dir = make_run_dir()

    n_videos = len(np.unique(video_ids)) if len(video_ids) > 0 else 0
    save_run_metadata(
        run_dir,
        extra={
            "n_samples": int(len(X)),
            "n_videos": n_videos,
            "class_distribution": (
                {CLASSES[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
                if len(y) > 0 else {}
            ),
            "model_type": args.model,
            "split_strategy": args.split,
            "augmentation": not args.no_augmentation,
            "lstm_units": args.lstm_units,
            "img_size": img_h,
            "fine_tune_layers_actual": ft_layers,
            "tag": args.tag,
        },
    )

    print("\n" + "=" * 50)
    print("RESUMO DO DATASET")
    print("=" * 50)
    print(f"Total de amostras: {len(X)}")
    print(f"Total de vídeos: {n_videos}")
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
    if args.split == "video":
        print(f"\nSplit por VÍDEO (GroupShuffleSplit) — sem data leakage")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=video_ids))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_vids = np.unique(video_ids[train_idx])
        test_vids = np.unique(video_ids[test_idx])
        print(f"  Vídeos treino: {len(train_vids)} | Vídeos teste: {len(test_vids)}")
    else:
        print(f"\nSplit por JANELA (legacy — pode haver data leakage)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste:  {len(X_test)} amostras")
    tr_unique, tr_counts = np.unique(y_train, return_counts=True)
    te_unique, te_counts = np.unique(y_test, return_counts=True)
    for idx, count in zip(tr_unique, tr_counts):
        print(f"  Treino {CLASSES[idx]}: {count}")
    for idx, count in zip(te_unique, te_counts):
        print(f"  Teste  {CLASSES[idx]}: {count}")

    # Dataset com augmentation
    if args.no_augmentation:
        print("Data augmentation DESATIVADO")
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        print("Data augmentation ATIVADO (flip + brilho)")
        train_ds = create_augmented_dataset(X_train, y_train, BATCH_SIZE)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Modelo
    print("\n" + "=" * 50)
    print(f"CONSTRUINDO MODELO: {args.model}")
    print("=" * 50)

    if args.model == "cnn_only":
        from src.model import build_cnn_only_model
        model = build_cnn_only_model(
            sequence_length=SEQUENCE_LENGTH,
            img_height=img_h,
            img_width=img_w,
            fine_tune_layers=ft_layers,
            learning_rate=LEARNING_RATE,
        )
    elif args.model == "cnn_lstm_frozen":
        model = build_cnn_lstm_model(
            sequence_length=SEQUENCE_LENGTH,
            img_height=img_h,
            img_width=img_w,
            fine_tune_layers=0,
            learning_rate=LEARNING_RATE,
            lstm_units=args.lstm_units,
        )
    else:
        model = build_cnn_lstm_model(
            sequence_length=SEQUENCE_LENGTH,
            img_height=img_h,
            img_width=img_w,
            fine_tune_layers=ft_layers,
            learning_rate=LEARNING_RATE,
            lstm_units=args.lstm_units,
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
