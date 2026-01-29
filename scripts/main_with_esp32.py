"""
Sistema de Detecção de Quedas com Integração ESP32
Versão do main.py que envia alertas para ESP32 quando queda é detectada
"""
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path

# Adicionar diretórios ao path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from esp32_interface import create_esp32_interface
from configs.config import (
    MODEL_PATH, IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH, CLASSES,
    ESP32_CONNECTION_TYPE, ESP32_PORT, ESP32_BAUDRATE
)

def main():
    print("="*60)
    print("SISTEMA DE DETECCAO DE QUEDAS COM ESP32")
    print("="*60)
    
    # Carregar Modelo
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo {MODEL_PATH} nao encontrado. Treine primeiro.")
        print("Execute: python train_model.py")
        return
    else:
        print("Carregando modelo...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo carregado!")
        except Exception as e:
            print(f"ERRO ao carregar modelo: {e}")
            return
    
    # Inicializar ESP32
    print("\nInicializando conexao com ESP32...")
    try:
        if ESP32_CONNECTION_TYPE == "serial":
            esp32 = create_esp32_interface(
                "serial",
                port=ESP32_PORT,
                baudrate=ESP32_BAUDRATE
            )
        else:  # MQTT
            esp32 = create_esp32_interface(
                "mqtt",
                broker=ESP32_BROKER,
                topic=ESP32_TOPIC
            )
        
        if not esp32.connected:
            print("AVISO: ESP32 nao conectado. Sistema continuara sem alertas.")
            esp32 = None
        else:
            print("ESP32 conectado com sucesso!")
    except Exception as e:
        print(f"ERRO ao conectar ESP32: {e}")
        print("Sistema continuara sem alertas.")
        esp32 = None
    
    # Inicializar câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Nao foi possivel abrir a camera.")
        if esp32:
            esp32.disconnect()
        return
    
    frames_queue = []
    last_fall_detected = False
    
    print("\nSistema iniciado. Pressione 'q' para sair.")
    print("Pressione 't' para enviar alerta de teste ao ESP32.\n")
    
    try:
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
            predicted_class = None
            confidence = None

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
                    color = (0, 0, 255)  # Vermelho
                else:
                    predicted_class = 'Normal'
                    confidence = 1 - prediction_prob
                    color = (0, 255, 0)  # Verde
                
                label = f"{predicted_class} ({confidence*100:.1f}%)"
                
                # Enviar alerta para ESP32 se queda detectada
                if predicted_class == 'Fall' and esp32 and esp32.connected:
                    # Evitar spam: só enviar se não estava detectando queda antes
                    if not last_fall_detected:
                        success = esp32.send_alert(
                            confidence=confidence,
                            metadata={
                                "frame_id": len(frames_queue),
                                "model": "CNN-LSTM"
                            }
                        )
                        if success:
                            print(f"[ALERTA] Queda detectada! Confianca: {confidence*100:.1f}%")
                    last_fall_detected = True
                else:
                    last_fall_detected = False
                
                # Visualização
                if predicted_class == 'Fall':
                    cv2.putText(frame, "ALERTA DE QUEDA!", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    if esp32 and esp32.connected:
                        cv2.putText(frame, "ESP32 ALERTA ATIVO", (50, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Mostrar na tela
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if esp32:
                status_text = "ESP32: CONECTADO" if esp32.connected else "ESP32: DESCONECTADO"
                status_color = (0, 255, 0) if esp32.connected else (0, 0, 255)
                cv2.putText(frame, status_text, (20, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.imshow("Fall Detection - CNN+LSTM + ESP32", frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t') and esp32:
                # Teste de alerta
                print("[TESTE] Enviando alerta de teste ao ESP32...")
                esp32.send_test_alert()
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
    finally:
        # Limpeza
        cap.release()
        cv2.destroyAllWindows()
        if esp32:
            esp32.disconnect()
        print("\nSistema finalizado.")

if __name__ == "__main__":
    main()
