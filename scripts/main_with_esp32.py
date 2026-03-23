"""
Sistema de Detecção de Quedas com Integração ESP32
Envia alertas via Serial/MQTT quando queda é detectada
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from esp32_interface import create_esp32_interface
from configs.config import (
    MODEL_PATH, TFLITE_MODEL_PATH, IMG_HEIGHT, IMG_WIDTH,
    SEQUENCE_LENGTH, CLASSES, INFERENCE_SKIP_FRAMES,
    CONFIDENCE_THRESHOLD, USE_TFLITE,
    ESP32_CONNECTION_TYPE, ESP32_PORT, ESP32_BAUDRATE,
)


def load_tflite_model(path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def predict_tflite(interpreter, input_details, output_details, input_data):
    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]


def main():
    print("=" * 60)
    print("SISTEMA DE DETECÇÃO DE QUEDAS COM ESP32")
    print("=" * 60)

    # --- Modelo ---
    use_tflite = USE_TFLITE and os.path.exists(TFLITE_MODEL_PATH)
    model = None
    tflite_ctx = None

    if use_tflite:
        print(f"Carregando modelo TFLite: {TFLITE_MODEL_PATH}")
        tflite_ctx = load_tflite_model(TFLITE_MODEL_PATH)
        print("Modelo TFLite carregado!")
    elif os.path.exists(MODEL_PATH):
        import tensorflow as tf
        print(f"Carregando modelo Keras: {MODEL_PATH}")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Modelo Keras carregado!")
    else:
        print(f"ERRO: Nenhum modelo encontrado.")
        print("Execute: python scripts/train_model.py")
        return

    # --- ESP32 ---
    print("\nInicializando conexão com ESP32...")
    esp32 = None
    try:
        if ESP32_CONNECTION_TYPE == "serial":
            esp32 = create_esp32_interface(
                "serial", port=ESP32_PORT, baudrate=ESP32_BAUDRATE
            )
        else:
            from configs.config import ESP32_BROKER, ESP32_TOPIC
            esp32 = create_esp32_interface(
                "mqtt", broker=ESP32_BROKER, topic=ESP32_TOPIC
            )

        if not esp32.connected:
            print("AVISO: ESP32 não conectado. Sistema continuará sem alertas.")
            esp32 = None
        else:
            print("ESP32 conectado com sucesso!")
    except Exception as e:
        print(f"ERRO ao conectar ESP32: {e}")
        print("Sistema continuará sem alertas.")
        esp32 = None

    # --- Câmera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a câmera.")
        if esp32:
            esp32.disconnect()
        return

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    last_fall_detected = False
    last_label = "Aguardando buffer..."
    last_color = (255, 255, 0)

    print(f"\nSistema iniciado (resolução {IMG_WIDTH}x{IMG_HEIGHT}, "
          f"predição a cada {INFERENCE_SKIP_FRAMES} frames).")
    print("Pressione 'q' para sair, 't' para teste de alerta.\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            normalized = resized / 255.0
            frames_queue.append(normalized)
            frame_count += 1

            predicted_class = None

            if len(frames_queue) == SEQUENCE_LENGTH and frame_count % INFERENCE_SKIP_FRAMES == 0:
                input_data = np.expand_dims(np.array(frames_queue), axis=0)

                if use_tflite:
                    prob = predict_tflite(*tflite_ctx, input_data)
                else:
                    prob = float(model(input_data, training=False)[0][0])

                if prob > CONFIDENCE_THRESHOLD:
                    predicted_class = 'Fall'
                    confidence = prob
                    last_color = (0, 0, 255)
                else:
                    predicted_class = 'Normal'
                    confidence = 1 - prob
                    last_color = (0, 255, 0)

                last_label = f"{predicted_class} ({confidence*100:.1f}%)"

                if predicted_class == 'Fall' and esp32 and esp32.connected:
                    if not last_fall_detected:
                        sent = esp32.send_alert(
                            confidence=confidence,
                            metadata={"frame_id": frame_count, "model": "CNN-LSTM"},
                        )
                        if sent:
                            print(f"[ALERTA] Queda detectada! Confiança: {confidence*100:.1f}%")
                    last_fall_detected = True
                else:
                    last_fall_detected = False

                if predicted_class == 'Fall':
                    cv2.putText(frame, "ALERTA DE QUEDA!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    if esp32 and esp32.connected:
                        cv2.putText(frame, "ESP32 ALERTA ATIVO", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.putText(frame, last_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
            if esp32:
                status_text = "ESP32: CONECTADO" if esp32.connected else "ESP32: DESCONECTADO"
                status_color = (0, 255, 0) if esp32.connected else (0, 0, 255)
                cv2.putText(frame, status_text, (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            cv2.imshow("Fall Detection - CNN+LSTM + ESP32", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t') and esp32:
                print("[TESTE] Enviando alerta de teste ao ESP32...")
                esp32.send_test_alert()

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if esp32:
            esp32.disconnect()
        print("\nSistema finalizado.")


if __name__ == "__main__":
    main()
