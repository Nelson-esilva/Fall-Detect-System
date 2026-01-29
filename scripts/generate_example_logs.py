"""
Script para gerar logs de exemplo (estrutura de logs de performance)
Útil para demonstrar a organização dos logs sem precisar treinar o modelo
"""
import os
import json
import csv
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS_DIR = 'logs'
CLASSES = ['Normal', 'Fall']

def make_run_dir() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(LOGS_DIR, f"run-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def generate_example_logs():
    """Gera logs de exemplo com dados simulados"""
    
    print("="*60)
    print("GERANDO LOGS DE EXEMPLO")
    print("="*60)
    
    run_dir = make_run_dir()
    print(f"\nDiretorio de logs: {run_dir}\n")
    
    # 1. Metadata
    metadata = {
        "timestamp": os.path.basename(run_dir).replace("run-", ""),
        "data_dir": "data/raw",
        "img_height": 224,
        "img_width": 224,
        "sequence_length": 20,
        "classes": CLASSES,
        "model_path": "models/fall_model_cnn_lstm.h5",
        "n_samples": 150,
        "class_distribution": {"Normal": 80, "Fall": 70},
        "train_samples": 120,
        "test_samples": 30,
        "epochs": 15,
        "batch_size": 4,
        "learning_rate": 0.001,
        "dropout": 0.5,
    }
    
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("[OK] run_metadata.json criado")
    
    # 2. History (simulado)
    epochs = 15
    history_data = {
        "loss": [0.6931, 0.5421, 0.4321, 0.3521, 0.2987, 0.2543, 0.2187, 0.1892, 0.1654, 0.1456, 0.1289, 0.1145, 0.1021, 0.0912, 0.0815],
        "accuracy": [0.5500, 0.6750, 0.7500, 0.8000, 0.8250, 0.8500, 0.8750, 0.8875, 0.9000, 0.9125, 0.9250, 0.9375, 0.9500, 0.9625, 0.9750],
        "val_loss": [0.7123, 0.5876, 0.4892, 0.4123, 0.3542, 0.3098, 0.2745, 0.2456, 0.2212, 0.2001, 0.1815, 0.1654, 0.1508, 0.1376, 0.1254],
        "val_accuracy": [0.5333, 0.6667, 0.7333, 0.8000, 0.8333, 0.8667, 0.9000, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333]
    }
    
    # JSON
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    print("[OK] history.json criado")
    
    # CSV
    with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(history_data.keys()))
        for e in range(epochs):
            writer.writerow([e + 1] + [history_data[k][e] for k in history_data.keys()])
    print("[OK] history.csv criado")
    
    # 3. Gráficos
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        ax[0].plot(history_data["loss"], label="train_loss", marker='o', markersize=3)
        ax[0].plot(history_data["val_loss"], label="val_loss", marker='s', markersize=3)
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Época")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        ax[1].plot(history_data["accuracy"], label="train_acc", marker='o', markersize=3)
        ax[1].plot(history_data["val_accuracy"], label="val_acc", marker='s', markersize=3)
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Época")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "curves.png"), dpi=200)
        plt.close(fig)
        print("[OK] curves.png criado")
    except Exception as e:
        print(f"[AVISO] Erro ao criar graficos: {e}")
    
    # 4. Métricas finais
    final_metrics = {
        "loss": 0.1254,
        "accuracy": 0.9333
    }
    
    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    print("[OK] final_metrics.json criado")
    
    # 5. Classification Report
    report = """              precision    recall  f1-score   support

      Normal       0.9500      0.9000    0.9243        20
        Fall       0.9167      0.9667    0.9412        30

    accuracy                           0.9333        50
   macro avg       0.9333      0.9333    0.9328        50
weighted avg       0.9333      0.9333    0.9328        50
"""
    
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("[OK] classification_report.txt criado")
    
    # 6. Confusion Matrix
    cm = [[18, 2], [1, 29]]  # [[TN, FP], [FN, TP]]
    
    with open(os.path.join(run_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(cm, f, ensure_ascii=False, indent=2)
    print("[OK] confusion_matrix.json criado")
    
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
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=200)
        plt.close(fig)
        print("[OK] confusion_matrix.png criado")
    except Exception as e:
        print(f"[AVISO] Erro ao criar matriz de confusao: {e}")
    
    # 7. Keras history CSV (simulado)
    with open(os.path.join(run_dir, "keras_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for e in range(epochs):
            writer.writerow([
                e + 1,
                history_data["loss"][e],
                history_data["accuracy"][e],
                history_data["val_loss"][e],
                history_data["val_accuracy"][e]
            ])
    print("[OK] keras_history.csv criado")
    
    print(f"\n{'='*60}")
    print(f"[OK] Logs de exemplo gerados com sucesso!")
    print(f"Localizacao: {run_dir}")
    print(f"{'='*60}\n")
    
    # Listar arquivos criados
    files = os.listdir(run_dir)
    print("Arquivos criados:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(run_dir, f))
        print(f"  - {f} ({size:,} bytes)")

if __name__ == "__main__":
    generate_example_logs()
