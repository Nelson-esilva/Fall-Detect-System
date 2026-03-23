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

# matplotlib é usado só para salvar gráficos/imagens (não abre janela)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Adicionar diretórios ao path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.model import build_cnn_lstm_model
from configs.config import (
    DATA_DIR, CLASSES, IMG_HEIGHT, IMG_WIDTH, 
    SEQUENCE_LENGTH, MODEL_PATH, LOGS_DIR
)


def make_run_dir() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(LOGS_DIR, f"run-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_history(history, run_dir: str):
    # JSON
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = os.path.join(run_dir, "history.csv")
    keys = list(history.history.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        epochs = len(history.history[keys[0]]) if keys else 0
        for e in range(epochs):
            writer.writerow([e + 1] + [history.history[k][e] for k in keys])

    # Plot
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
        print(f"⚠️  Aviso: falha ao salvar curvas: {e}")


def save_eval_artifacts(model, X_test, y_test, run_dir: str):
    # Predições
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob > 0.5).astype(np.int32)

    # Métricas / relatório
    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4, zero_division=0)
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(run_dir, "confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, ensure_ascii=False, indent=2)

    # Plot confusion matrix
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
        print(f"⚠️  Aviso: falha ao salvar matriz de confusão: {e}")

    # Salvar probabilidades (leve e útil)
    try:
        np.save(os.path.join(run_dir, "y_test.npy"), y_test)
        np.save(os.path.join(run_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(run_dir, "y_prob.npy"), y_prob)
    except Exception as e:
        print(f"⚠️  Aviso: falha ao salvar arrays: {e}")


def save_run_metadata(run_dir: str, extra: dict):
    meta = {
        "timestamp": os.path.basename(run_dir).replace("run-", ""),
        "data_dir": str(DATA_DIR),
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "sequence_length": SEQUENCE_LENGTH,
        "classes": CLASSES,
        "model_path": str(MODEL_PATH),
    }
    meta.update(extra or {})
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_data():
    """
    Lê os arquivos de vídeo das pastas e cria o dataset X, y.
    Para vídeos longos, extrai múltiplas sequências (data augmentation).
    X shape: (n_samples, 20, 224, 224, 3)
    y shape: (n_samples,)
    """
    features = []
    labels = []
    
    print("="*50)
    print("CARREGANDO DATASET")
    print("="*50)
    
    for class_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  AVISO: Pasta '{class_dir}' não encontrada!")
            continue
        
        files = [f for f in os.listdir(class_dir) if f.endswith(('.avi', '.mp4'))]
        print(f"\n📂 Classe '{class_name}': {len(files)} vídeos encontrados")
        
        if len(files) == 0:
            print(f"   ❌ ERRO: Nenhum vídeo na pasta {class_name}!")
            continue
            
        for file_name in files:
            video_path = os.path.join(class_dir, file_name)
            cap = cv2.VideoCapture(video_path)
            
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                frame = frame / 255.0  # Normalizar
                all_frames.append(frame)
            
            cap.release()
            
            # Extrair múltiplas sequências do vídeo (sliding window)
            # Isso aumenta a quantidade de dados de treino
            if len(all_frames) >= SEQUENCE_LENGTH:
                # Passo de 10 frames para criar sobreposição
                step = max(1, SEQUENCE_LENGTH // 2)
                for start in range(0, len(all_frames) - SEQUENCE_LENGTH + 1, step):
                    sequence = all_frames[start:start + SEQUENCE_LENGTH]
                    features.append(sequence)
                    labels.append(class_index)
                
                print(f"   ✅ {file_name}: {len(all_frames)} frames → {(len(all_frames) - SEQUENCE_LENGTH) // step + 1} amostras")
            else:
                print(f"   ⚠️  {file_name}: Apenas {len(all_frames)} frames (mínimo: {SEQUENCE_LENGTH})")

    return np.array(features), np.array(labels)

# 1. Carregar Dados
X, y = load_data()

run_dir = make_run_dir()
save_run_metadata(
    run_dir,
    extra={
        "n_samples": int(len(X)),
        "class_distribution": {CLASSES[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    },
)

print("\n" + "="*50)
print("RESUMO DO DATASET")
print("="*50)
print(f"Total de amostras: {len(X)}")

if len(X) == 0:
    print("\n❌ ERRO CRÍTICO: Nenhum dado encontrado!")
    print("Verifique se você:")
    print("  1. Baixou os vídeos ADL (adl-*-cam0-rgb.zip) do UR Fall")
    print("  2. Rodou 'python prepare_ur_fall.py'")
    exit()

# Contar amostras por classe
unique, counts = np.unique(y, return_counts=True)
for idx, count in zip(unique, counts):
    print(f"  - {CLASSES[idx]}: {count} amostras")

# Verificar desbalanceamento crítico
if len(unique) < 2:
    print("\n❌ ERRO CRÍTICO: Apenas uma classe encontrada!")
    print("O modelo precisa de exemplos de AMBAS as classes (Normal e Fall).")
    print("Baixe os vídeos ADL do UR Fall Dataset.")
    exit()

# 2. Dividir Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {len(X_train)} amostras")
print(f"Teste:  {len(X_test)} amostras")

# 3. Construir Modelo
print("\n" + "="*50)
print("CONSTRUINDO MODELO")
print("="*50)
model = build_cnn_lstm_model()
model.summary()

# 4. Treinar
print("\n" + "="*50)
print("INICIANDO TREINAMENTO")
print("="*50)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "keras_history.csv"), append=False),
]

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

save_history(history, run_dir)

# 5. Avaliação Final
print("\n" + "="*50)
print("AVALIAÇÃO FINAL")
print("="*50)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no conjunto de teste: {accuracy*100:.2f}%")
print(f"Loss no conjunto de teste: {loss:.4f}")
print(f"\n✅ Modelo salvo em: {MODEL_PATH}")

with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"loss": float(loss), "accuracy": float(accuracy)}, f, ensure_ascii=False, indent=2)

save_eval_artifacts(model, X_test, y_test, run_dir)
print(f"\n📁 Logs salvos em: {run_dir}")
