import cv2
import os
import time

# Configurações
OUTPUT_DIR = 'data/raw'
CLASSES = ['Normal', 'Fall']
SEQUENCE_LENGTH = 20  # Número de frames por vídeo
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Criar pastas
for c in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

cap = cv2.VideoCapture(0)

print(f"========================================")
print(f"COLETOR DE VÍDEOS PARA CNN-LSTM")
print(f"Resolução de salvamento: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Pressione 'n' para gravar sequência NORMAL")
print(f"Pressione 'f' para gravar sequência QUEDA")
print(f"Pressione 'q' para sair")
print(f"========================================")

recording = False
current_class = None
frames_buffer = []
video_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Redimensionar para visualização e processamento
    display_frame = frame.copy()
    
    # Se estiver gravando
    if recording:
        # Resize para o tamanho que a rede espera
        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frames_buffer.append(resized)
        
        # Indicador visual
        cv2.circle(display_frame, (30, 30), 20, (0, 0, 255), -1)
        cv2.putText(display_frame, f"GRAVANDO {current_class}: {len(frames_buffer)}/{SEQUENCE_LENGTH}", 
                   (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Se completou a sequência
        if len(frames_buffer) == SEQUENCE_LENGTH:
            # Salvar vídeo
            timestamp = int(time.time())
            filename = os.path.join(OUTPUT_DIR, current_class, f"{current_class}_{timestamp}.avi")
            
            # Configurar Writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 20.0, (IMG_WIDTH, IMG_HEIGHT))
            
            for f_out in frames_buffer:
                out.write(f_out)
            out.release()
            
            print(f"Vídeo salvo: {filename}")
            
            # Reset
            frames_buffer = []
            recording = False
            current_class = None

    cv2.imshow("Coletor de Videos", display_frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    
    # Iniciar gravação se não estiver gravando
    if not recording:
        if key == ord('n'):
            recording = True
            current_class = 'Normal'
            frames_buffer = []
            print("Gravando NORMAL...")
        elif key == ord('f'):
            recording = True
            current_class = 'Fall'
            frames_buffer = []
            print("Gravando QUEDA...")

cap.release()
cv2.destroyAllWindows()

