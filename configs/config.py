"""
Configurações centralizadas do projeto
"""
import os
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
IMG_HEIGHT, IMG_WIDTH = 224, 224
SEQUENCE_LENGTH = 20
CLASSES = ['Normal', 'Fall']

# Configurações ESP32
ESP32_CONNECTION_TYPE = "serial"  # "serial" ou "mqtt"
ESP32_PORT = "COM3"  # Ajuste para sua porta
ESP32_BAUDRATE = 115200

# Para MQTT (descomente se usar):
# ESP32_BROKER = "192.168.1.100"
# ESP32_TOPIC = "fall_detection/alerts"

# Criar diretórios se não existirem
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
