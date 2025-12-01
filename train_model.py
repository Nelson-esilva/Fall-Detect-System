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
MODEL_PATH = 'models/fall_model_cnn_lstm.h5' # Formato Keras

def load_data():
    """
    Lê os arquivos de vídeo das pastas e cria o dataset X, y.
    X shape: (n_videos, 20, 224, 224, 3)
    y shape: (n_videos,)
    """
    features = []
    labels = []
    
    print("Carregando vídeos...")
    
    for class_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        files = os.listdir(class_dir)
        for file_name in files:
            video_path = os.path.join(class_dir, file_name)
            cap = cv2.VideoCapture(video_path)
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Garantir tamanho correto (já deve estar, mas por segurança)
                frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                # Normalizar pixels (0 a 1) - Importante para Redes Neurais
                frame = frame / 255.0
                frames.append(frame)
            
            cap.release()
            
            # Pegar apenas vídeos que tenham frames suficientes
            # Se tiver mais, corta. Se tiver menos, ignora (ou padding)
            if len(frames) >= SEQUENCE_LENGTH:
                features.append(frames[:SEQUENCE_LENGTH])
                labels.append(class_index) # 0 = Normal, 1 = Fall

    return np.array(features), np.array(labels)

# 1. Carregar Dados
X, y = load_data()

print(f"Dados carregados: {len(X)} amostras.")
if len(X) == 0:
    print("ERRO: Nenhum vídeo encontrado. Rode 'collect_videos.py' primeiro.")
    exit()

# 2. Dividir Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Construir Modelo
model = build_cnn_lstm_model()
model.summary()

# 4. Treinar
print("Iniciando treinamento...")
# Checkpoint para salvar o melhor modelo durante o treino
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'
)

history = model.fit(
    X_train, y_train,
    epochs=10,            # Pode aumentar se tiver GPU
    batch_size=4,         # Pequeno porque vídeo gasta muita memória RAM
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

print(f"Modelo salvo em {MODEL_PATH}")
