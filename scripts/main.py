import cv2
import numpy as np
import os
import sys
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    MODEL_PATH, TFLITE_MODEL_PATH, IMG_HEIGHT, IMG_WIDTH,
    SEQUENCE_LENGTH, CLASSES, INFERENCE_SKIP_FRAMES,
    CONFIDENCE_THRESHOLD, USE_TFLITE,
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
        print(f"ERRO: Nenhum modelo encontrado em {MODEL_PATH} ou {TFLITE_MODEL_PATH}")
        print("Treine primeiro com: python scripts/train_model.py")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a câmera.")
        return

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    last_label = "Aguardando buffer..."
    last_color = (255, 255, 0)

    print(f"Sistema iniciado (resolução {IMG_WIDTH}x{IMG_HEIGHT}, "
          f"predição a cada {INFERENCE_SKIP_FRAMES} frames). Pressione 'q' para sair.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        normalized = resized / 255.0
        frames_queue.append(normalized)
        frame_count += 1

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

            if predicted_class == 'Fall':
                cv2.putText(frame, "ALERTA DE QUEDA!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.putText(frame, last_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
        cv2.imshow("Fall Detection - CNN+LSTM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
