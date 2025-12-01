import cv2
import numpy as np
import tensorflow as tf
import os

# Configurações
MODEL_PATH = 'models/fall_model_cnn_lstm.h5'
IMG_HEIGHT, IMG_WIDTH = 224, 224
SEQUENCE_LENGTH = 20
CLASSES = ['Normal', 'Fall']

# Carregar Modelo
if not os.path.exists(MODEL_PATH):
    print(f"ERRO: Modelo {MODEL_PATH} não encontrado. Treine primeiro.")
    # Criar modelo dummy só para não crashar se o usuário quiser testar a câmera
    model = None
else:
    print("Carregando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado!")

cap = cv2.VideoCapture(0)
frames_queue = [] # Fila para guardar os últimos frames

print("Iniciando Sistema. Pressione 'q' para sair.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Pré-processamento para o modelo
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    normalized_frame = resized_frame / 255.0
    
    # Adicionar à fila
    frames_queue.append(normalized_frame)
    
    # Manter tamanho fixo da fila
    if len(frames_queue) > SEQUENCE_LENGTH:
        frames_queue.pop(0)

    label = "Aguardando buffer..."
    color = (255, 255, 0)

    # Se a fila estiver cheia, podemos fazer predição
    if len(frames_queue) == SEQUENCE_LENGTH and model is not None:
        # Preparar batch: (1, 20, 224, 224, 3)
        input_data = np.expand_dims(np.array(frames_queue), axis=0)
        
        # Predição
        prediction_prob = model.predict(input_data, verbose=0)[0][0]
        
        # Limiar de decisão (0.5)
        if prediction_prob > 0.5:
            predicted_class = 'Fall'
            confidence = prediction_prob
            color = (0, 0, 255) # Vermelho
        else:
            predicted_class = 'Normal'
            confidence = 1 - prediction_prob
            color = (0, 255, 0) # Verde
            
        label = f"{predicted_class} ({confidence*100:.1f}%)"
        
        if predicted_class == 'Fall':
            cv2.putText(frame, "ALERTA DE QUEDA!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Mostrar na tela
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fall Detection - CNN+LSTM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
