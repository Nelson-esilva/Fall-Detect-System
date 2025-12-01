import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.model import build_cnn_lstm_model

# Configurações
DATA_DIR = 'data/raw'
CLASSES = ['Normal', 'Fall']
IMG_HEIGHT, IMG_WIDTH = 224, 224
SEQUENCE_LENGTH = 20
MODEL_PATH = 'models/fall_model_cnn_lstm.h5'

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
    )
]

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# 5. Avaliação Final
print("\n" + "="*50)
print("AVALIAÇÃO FINAL")
print("="*50)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no conjunto de teste: {accuracy*100:.2f}%")
print(f"Loss no conjunto de teste: {loss:.4f}")
print(f"\n✅ Modelo salvo em: {MODEL_PATH}")
