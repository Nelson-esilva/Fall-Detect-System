import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configurações do Relatório
SEQUENCE_LENGTH = 20   # Tamanho da janela temporal (20 frames)
IMG_HEIGHT = 224       # Altura padrão da MobileNetV2
IMG_WIDTH = 224        # Largura padrão
CHANNELS = 3           # RGB

def build_cnn_lstm_model():
    """
    Constrói o modelo híbrido CNN-LSTM para detecção de quedas.
    Baseado na arquitetura descrita no Relatório de Setembro/2025.
    """
    print("Construindo o modelo CNN-LSTM...")
    
    # Definir input: uma sequência de frames
    # Shape: (Batch, 20 frames, 224, 224, 3)
    input_shape = (SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    inputs = keras.Input(shape=input_shape)

    # --- 1. Extrator de Características Visuais (CNN) ---
    # Usamos a MobileNetV2 pré-treinada no ImageNet para "entender" o que tem na imagem
    base_model = keras.applications.MobileNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    )
    
    # Congelamos a base para não desaprender o que ela já sabe (Transfer Learning)
    base_model.trainable = False

    # TimeDistributed permite aplicar a MESMA CNN em cada um dos 20 frames individualmente
    cnn_output = layers.TimeDistributed(base_model)(inputs)
    
    # Reduzir a dimensionalidade espacial de cada frame (de 7x7x1280 para 1280)
    cnn_output = layers.TimeDistributed(layers.GlobalAveragePooling2D())(cnn_output)

    # --- 2. Analisador de Sequência Temporal (LSTM) ---
    # A LSTM olha para a sequência de 20 vetores de características e aprende o movimento
    x = layers.LSTM(64, return_sequences=False)(cnn_output)
    
    # Camada de Dropout para evitar Overfitting (o modelo decorar os dados)
    x = layers.Dropout(0.5)(x)

    # --- 3. Classificação ---
    # Saída binária: 0 (Normal/ADL) ou 1 (Fall/Queda)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    
    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

