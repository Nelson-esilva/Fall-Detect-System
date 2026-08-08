"""
Configurações centralizadas do projeto
"""
from pathlib import Path

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# Diretórios
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
UR_FALL_DIR = PROJECT_ROOT / 'UR_Fall_Downloads'

# Configurações do modelo
MODEL_PATH = MODELS_DIR / 'fall_model_cnn_lstm.h5'
TFLITE_MODEL_PATH = MODELS_DIR / 'fall_model_cnn_lstm.tflite'
IMG_HEIGHT, IMG_WIDTH = 128, 128
SEQUENCE_LENGTH = 20
CLASSES = ['Normal', 'Fall']

# Inferência em tempo real
INFERENCE_SKIP_FRAMES = 5       # predizer a cada N frames (1 = todo frame)
CONFIDENCE_THRESHOLD = 0.5      # limiar para considerar queda
USE_TFLITE = True               # usar TFLite quando disponível (mais rápido em CPU)

# Treinamento
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
FINE_TUNE_LAYERS = 30           # camadas da MobileNetV2 a descongelar (0 = todas congeladas)
USE_MIXED_PRECISION = False     # float16 — desativado (TimeDistributed trava XLA no T4)

# Configurações ESP32
ESP32_CONNECTION_TYPE = "serial"  # "serial" ou "mqtt"
ESP32_PORT = "COM3"               # Ajuste para sua porta
ESP32_BAUDRATE = 115200

# MQTT (descomente se usar)
# ESP32_BROKER = "192.168.1.100"
# ESP32_TOPIC = "fall_detection/alerts"

# Criar diretórios se não existirem
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
