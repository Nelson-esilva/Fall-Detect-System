try:
    import cv2
except ImportError:
    print("ERRO: opencv-python nao instalado.")
    print("Execute: pip install opencv-python")
    exit(1)

import numpy as np
import os
import glob
import time
from pathlib import Path

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("AVISO: TensorFlow nao disponivel. Usando modo de simulacao apenas.")

# Configurações
MODEL_PATH = 'models/fall_model_cnn_lstm.h5'
UR_FALL_DIR = 'UR_Fall_Downloads'
IMG_HEIGHT, IMG_WIDTH = 224, 224
SEQUENCE_LENGTH = 20
CLASSES = ['Normal', 'Fall']
FPS_SIMULATION = 10  # FPS para a simulação (mais lento para visualização)

def find_fall_videos():
    """Encontra pastas com sequências de queda no dataset UR Fall"""
    fall_folders = []
    if os.path.exists(UR_FALL_DIR):
        for item in os.listdir(UR_FALL_DIR):
            item_path = os.path.join(UR_FALL_DIR, item)
            if os.path.isdir(item_path) and 'fall' in item.lower():
                # Verificar se tem imagens PNG
                png_files = glob.glob(os.path.join(item_path, '**', '*.png'), recursive=True)
                if len(png_files) >= SEQUENCE_LENGTH:
                    fall_folders.append((item, sorted(png_files)))
    return fall_folders

def load_frames_from_folder(image_paths):
    """Carrega e normaliza frames de uma lista de caminhos de imagens"""
    frames = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            normalized = resized / 255.0
            frames.append((resized, normalized))  # (original para display, normalized para modelo)
    return frames

def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    """Desenha retângulo com cantos arredondados"""
    x1, y1 = pt1
    x2, y2 = pt2
    # Cantos
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    # Bordas
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    # Borda externa
    if thickness > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_info_panel(frame, buffer_size, prediction=None, confidence=None, frame_count=0, total_frames=0, 
                    confidence_history=None, frames_queue=None):
    """Desenha painel de informações moderno sobre a simulação"""
    h, w = frame.shape[:2]
    
    # Painel lateral direito (mais espaço)
    panel_width = 380
    panel_x = w - panel_width
    
    # Fundo do painel com gradiente
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (15, 15, 25), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    
    # Linha divisória
    cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 60, 80), 2)
    
    y_offset = 20
    line_height = 28
    
    # Título com fundo destacado
    title_bg_y = y_offset - 15
    cv2.rectangle(frame, (panel_x + 10, title_bg_y), (w - 10, y_offset + 25), (30, 50, 100), -1)
    cv2.rectangle(frame, (panel_x + 10, title_bg_y), (w - 10, y_offset + 25), (60, 100, 200), 2)
    cv2.putText(frame, "SISTEMA DE DETECCAO", (panel_x + 15, y_offset + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "DE QUEDAS", (panel_x + 15, y_offset + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 200, 255), 2)
    y_offset += 50
    
    # Seção: Status do Buffer
    section_y = y_offset
    cv2.putText(frame, "BUFFER DE FRAMES", (panel_x + 15, section_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 25
    
    buffer_percent = (buffer_size / SEQUENCE_LENGTH) * 100
    cv2.putText(frame, f"{buffer_size} / {SEQUENCE_LENGTH} frames", (panel_x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    y_offset += 25
    
    # Barra de progresso moderna
    bar_x, bar_y = panel_x + 15, y_offset
    bar_width = panel_width - 50
    bar_height = 22
    bar_radius = 5
    
    # Fundo da barra
    draw_rounded_rect(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (40, 40, 50), -1, bar_radius)
    
    # Preenchimento da barra
    fill_width = int((buffer_size / SEQUENCE_LENGTH) * bar_width)
    if fill_width > 0:
        if buffer_size == SEQUENCE_LENGTH:
            fill_color = (0, 200, 100)  # Verde quando completo
        else:
            fill_color = (0, 150, 255)  # Azul durante carregamento
        draw_rounded_rect(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                          fill_color, -1, bar_radius)
    
    # Texto da porcentagem
    percent_text = f"{buffer_percent:.0f}%"
    text_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = bar_x + (bar_width - text_size[0]) // 2
    cv2.putText(frame, percent_text, (text_x, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 40
    
    # Seção: Progresso do Vídeo
    if total_frames > 0:
        progress = (frame_count / total_frames) * 100
        cv2.putText(frame, "PROGRESSO DO VIDEO", (panel_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += 25
        cv2.putText(frame, f"Frame {frame_count}/{total_frames} ({progress:.1f}%)", 
                    (panel_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        # Barra de progresso do vídeo
        video_bar_x, video_bar_y = panel_x + 15, y_offset
        video_bar_width = panel_width - 50
        video_bar_height = 8
        cv2.rectangle(frame, (video_bar_x, video_bar_y), 
                     (video_bar_x + video_bar_width, video_bar_y + video_bar_height), 
                     (50, 50, 60), -1)
        video_fill = int((frame_count / total_frames) * video_bar_width)
        if video_fill > 0:
            cv2.rectangle(frame, (video_bar_x, video_bar_y), 
                         (video_bar_x + video_fill, video_bar_y + video_bar_height), 
                         (100, 150, 255), -1)
        y_offset += 30
    
    # Seção: Status da Predição
    cv2.putText(frame, "STATUS DA DETECCAO", (panel_x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 30
    
    if buffer_size == SEQUENCE_LENGTH:
        if prediction is not None:
            # Card de predição
            card_x, card_y = panel_x + 15, y_offset - 5
            card_w, card_h = panel_width - 50, 80
            
            if prediction == 'Fall':
                card_color = (0, 0, 150)  # Vermelho escuro
                border_color = (0, 0, 255)
                text_color = (0, 150, 255)
            else:
                card_color = (0, 100, 0)  # Verde escuro
                border_color = (0, 255, 0)
                text_color = (150, 255, 150)
            
            draw_rounded_rect(frame, (card_x, card_y), (card_x + card_w, card_y + card_h),
                             card_color, -1, 8)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h),
                         border_color, 2)
            
            # Texto da predição
            pred_text = "QUEDA DETECTADA!" if prediction == 'Fall' else "ATIVIDADE NORMAL"
            text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = card_x + (card_w - text_size[0]) // 2
            cv2.putText(frame, pred_text, (text_x, card_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            if confidence is not None:
                conf_text = f"Confianca: {confidence*100:.1f}%"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
                conf_x = card_x + (card_w - conf_size[0]) // 2
                cv2.putText(frame, conf_text, (conf_x, card_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                
                # Barra de confiança
                conf_bar_x, conf_bar_y = card_x + 10, card_y + 65
                conf_bar_w = card_w - 20
                conf_bar_h = 6
                cv2.rectangle(frame, (conf_bar_x, conf_bar_y), 
                             (conf_bar_x + conf_bar_w, conf_bar_y + conf_bar_h), 
                             (30, 30, 30), -1)
                conf_fill = int(confidence * conf_bar_w)
                if conf_fill > 0:
                    cv2.rectangle(frame, (conf_bar_x, conf_bar_y), 
                                 (conf_bar_x + conf_fill, conf_bar_y + conf_bar_h), 
                                 border_color, -1)
            
            y_offset += card_h + 15
        else:
            cv2.putText(frame, "Processando...", (panel_x + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            y_offset += 30
    else:
        waiting_text = f"Aguardando buffer... ({buffer_size}/{SEQUENCE_LENGTH})"
        cv2.putText(frame, waiting_text, (panel_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        y_offset += 30
    
    # Seção: Miniaturas dos Frames
    if frames_queue and len(frames_queue) > 0:
        y_offset += 10
        cv2.putText(frame, "ULTIMOS FRAMES", (panel_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += 25
        
        # Mostrar miniaturas reais dos últimos frames
        thumb_size = 45  # Reduzido para garantir que caiba
        thumb_spacing = 50
        num_thumbs = min(6, len(frames_queue))
        start_x = panel_x + 15
        
        # Verificar se há espaço suficiente antes de desenhar
        required_height = y_offset + ((num_thumbs // 3 + (1 if num_thumbs % 3 > 0 else 0)) * thumb_spacing)
        
        if required_height < h - 80:  # Deixar espaço para controles
            for i in range(num_thumbs):
                idx = len(frames_queue) - num_thumbs + i
                if idx >= 0:
                    thumb_x = start_x + (i % 3) * thumb_spacing
                    thumb_y = y_offset + (i // 3) * thumb_spacing
                    
                    # Verificar limites antes de colar
                    if (thumb_y + thumb_size < h and thumb_x + thumb_size < w and 
                        thumb_y >= 0 and thumb_x >= 0):
                        
                        # Redimensionar frame para miniatura
                        thumb_frame = cv2.resize(frames_queue[idx] * 255, (thumb_size, thumb_size))
                        thumb_frame = thumb_frame.astype(np.uint8)
                        
                        # Verificar se o formato está correto (BGR)
                        if len(thumb_frame.shape) == 3 and thumb_frame.shape[2] == 3:
                            # Desenhar miniatura apenas se couber
                            if (thumb_y + thumb_size <= frame.shape[0] and 
                                thumb_x + thumb_size <= frame.shape[1]):
                                frame[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size] = thumb_frame
                                
                                # Borda
                                border_color = (0, 255, 255) if i == num_thumbs - 1 else (100, 100, 100)
                                cv2.rectangle(frame, (thumb_x, thumb_y), 
                                             (thumb_x + thumb_size, thumb_y + thumb_size), 
                                             border_color, 2)
        
        y_offset += ((num_thumbs // 3 + (1 if num_thumbs % 3 > 0 else 0)) * thumb_spacing)
    
    # Gráfico de confiança ao longo do tempo (se disponível e houver espaço)
    if confidence_history and len(confidence_history) > 1:
        graph_height = 50
        required_graph_space = y_offset + 50 + graph_height
        
        if required_graph_space < h - 80:  # Verificar se há espaço para o gráfico
            y_offset += 15
            cv2.putText(frame, "HISTORICO DE CONFIANCA", (panel_x + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_offset += 25
            
            graph_x, graph_y = panel_x + 15, y_offset
            graph_w, graph_h = panel_width - 50, graph_height
            
            # Verificar limites
            if graph_y + graph_h < h - 80:
                # Fundo do gráfico
                cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                             (25, 25, 35), -1)
                cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                             (80, 80, 100), 1)
                
                # Desenhar linha de confiança
                if len(confidence_history) > 1:
                    points = []
                    history_slice = confidence_history[-50:]  # Últimos 50 pontos
                    for i, conf in enumerate(history_slice):
                        x = graph_x + int((i / max(1, len(history_slice) - 1)) * graph_w)
                        y = graph_y + graph_h - int(conf * graph_h)
                        # Garantir que y está dentro dos limites
                        y = max(graph_y, min(graph_y + graph_h, y))
                        points.append((x, y))
                    
                    for i in range(len(points) - 1):
                        if (points[i][0] < w and points[i+1][0] < w and
                            points[i][1] < h and points[i+1][1] < h):
                            hist_idx = len(confidence_history) - len(history_slice) + i
                            if hist_idx >= 0 and hist_idx < len(confidence_history):
                                color = (0, 0, 255) if confidence_history[hist_idx] > 0.5 else (0, 255, 0)
                                cv2.line(frame, points[i], points[i+1], color, 2)
                    
                    # Linha de threshold
                    threshold_y = graph_y + graph_h - int(0.5 * graph_h)
                    if threshold_y < h:
                        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y),
                                (150, 150, 150), 1)
    
    # Controles na parte inferior
    controls_y = h - 60
    cv2.putText(frame, "CONTROLES:", (panel_x + 15, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    controls_y += 20
    cv2.putText(frame, "Q - Sair | P - Pausar | R - Reiniciar", 
                (panel_x + 15, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def simulate_fall_detection():
    """Simula o processo de detecção de queda usando vídeos do dataset"""
    
    print("="*60)
    print("SIMULACAO DE DETECCAO DE QUEDA")
    print("="*60)
    
    # Carregar modelo se existir
    model = None
    use_real_model = False
    if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            print(f"Carregando modelo de {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo carregado com sucesso!")
            use_real_model = True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            print("Usando modo de simulacao (sem modelo treinado)")
    else:
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow nao disponivel. Usando modo de simulacao.")
        elif not os.path.exists(MODEL_PATH):
            print(f"Modelo nao encontrado em {MODEL_PATH}")
            print("Usando modo de simulacao (sem modelo treinado)")
            print("Para usar modelo real, execute primeiro: python train_model.py")
    
    # Encontrar vídeos de queda
    fall_videos = find_fall_videos()
    if not fall_videos:
        print(f"\nERRO: Nenhum video de queda encontrado em {UR_FALL_DIR}")
        print("Execute primeiro: python prepare_ur_fall.py")
        return
    
    print(f"\nEncontrados {len(fall_videos)} videos de queda")
    print("Videos disponiveis:")
    for i, (folder_name, paths) in enumerate(fall_videos[:10]):  # Mostrar apenas os 10 primeiros
        print(f"  [{i+1}] {folder_name} ({len(paths)} frames)")
    
    # Selecionar primeiro vídeo automaticamente
    selected_idx = 0
    print(f"\nUsando automaticamente: {fall_videos[selected_idx][0]}")
    
    folder_name, image_paths = fall_videos[selected_idx]
    print(f"\nProcessando: {folder_name}")
    print(f"Total de frames: {len(image_paths)}")
    
    # Carregar frames
    print("Carregando frames...")
    frames_data = load_frames_from_folder(image_paths)
    if len(frames_data) < SEQUENCE_LENGTH:
        print(f"ERRO: Video muito curto ({len(frames_data)} frames). Necessario pelo menos {SEQUENCE_LENGTH} frames.")
        return
    
    print(f"Frames carregados: {len(frames_data)}")
    print("\nIniciando simulacao...")
    print("Pressione 'q' para sair, 'p' para pausar, 'r' para reiniciar")
    
    # Buffer de frames normalizados
    frames_queue = []
    frame_idx = 0
    paused = False
    confidence_history = []  # Histórico de confiança para gráfico
    
    while frame_idx < len(frames_data):
        if not paused:
            # Obter frame atual
            frame_original, frame_normalized = frames_data[frame_idx]
            
            # Adicionar ao buffer
            frames_queue.append(frame_normalized)
            
            # Manter tamanho fixo
            if len(frames_queue) > SEQUENCE_LENGTH:
                frames_queue.pop(0)
            
            # Preparar frame para display (redimensionar para melhor visualização)
            display_frame = cv2.resize(frame_original, (640, 480))  # Tamanho fixo para display
            
            # Fazer predição se buffer estiver completo
            prediction = None
            confidence = None
            
            if len(frames_queue) == SEQUENCE_LENGTH:
                if use_real_model and model is not None:
                    # Predição real com modelo
                    input_data = np.expand_dims(np.array(frames_queue), axis=0)
                    prediction_prob = model.predict(input_data, verbose=0)[0][0]
                    
                    if prediction_prob > 0.5:
                        prediction = 'Fall'
                        confidence = prediction_prob
                    else:
                        prediction = 'Normal'
                        confidence = 1 - prediction_prob
                else:
                    # Simulação: para vídeos de queda, simular detecção após alguns frames
                    # Quedas geralmente ocorrem no meio/final da sequência
                    progress = frame_idx / len(frames_data)
                    if progress > 0.3:  # Após 30% do vídeo, começar a detectar queda
                        # Simular probabilidade crescente de queda
                        sim_prob = min(0.95, 0.5 + (progress - 0.3) * 0.6)
                        prediction = 'Fall'
                        confidence = sim_prob
                    else:
                        prediction = 'Normal'
                        confidence = 0.3
            
            # Atualizar histórico de confiança
            if confidence is not None:
                confidence_history.append(confidence)
                if len(confidence_history) > 100:  # Manter apenas últimos 100 pontos
                    confidence_history.pop(0)
            
            # Preparar frames para miniatura (últimos frames normalizados convertidos)
            frames_for_thumb = []
            if len(frames_queue) > 0:
                for f in frames_queue[-6:]:  # Últimos 6 frames
                    frames_for_thumb.append(f)
            
            # Desenhar informações
            display_frame = draw_info_panel(
                display_frame, 
                len(frames_queue), 
                prediction, 
                confidence,
                frame_idx,
                len(frames_data),
                confidence_history,
                frames_for_thumb
            )
            
            # Alerta visual se queda detectada
            if prediction == 'Fall':
                # Borda vermelha piscante com efeito de pulso
                pulse = int(time.time() * 3) % 2
                thickness = 8 if pulse == 0 else 5
                border_color = (0, 0, 255) if pulse == 0 else (0, 100, 255)
                cv2.rectangle(display_frame, (0, 0), 
                             (display_frame.shape[1]-1, display_frame.shape[0]-1),
                             border_color, thickness)
                
                # Overlay vermelho semi-transparente
                overlay_red = display_frame.copy()
                cv2.rectangle(overlay_red, (0, 0), 
                              (display_frame.shape[1], display_frame.shape[0]),
                              (0, 0, 150), -1)
                display_frame = cv2.addWeighted(display_frame, 0.85, overlay_red, 0.15, 0)
                
                # Banner de alerta no topo
                banner_h = 80
                banner_overlay = display_frame.copy()
                cv2.rectangle(banner_overlay, (0, 0), 
                             (display_frame.shape[1], banner_h),
                             (0, 0, 200), -1)
                display_frame = cv2.addWeighted(display_frame, 0.7, banner_overlay, 0.3, 0)
                
                # Texto de alerta principal
                text = "ALERTA DE QUEDA DETECTADA!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)[0]
                text_x = (display_frame.shape[1] - text_size[0]) // 2
                text_y = 50
                
                # Sombra do texto
                cv2.putText(display_frame, text, (text_x + 3, text_y + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
                # Texto principal
                cv2.putText(display_frame, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 4)
                
                # Ícone de alerta (triângulo de aviso)
                center_x = display_frame.shape[1] // 2
                center_y = display_frame.shape[0] // 2
                
                # Círculos concêntricos pulsantes
                for i in range(3):
                    radius = 40 + int(time.time() * 8) % 30 + (i * 15)
                    alpha = max(0, 255 - (i * 80))
                    color = (0, 0, min(255, alpha))
                    cv2.circle(display_frame, (center_x, center_y), radius, color, 2)
                
                # Triângulo de alerta
                triangle_size = 30
                pts = np.array([
                    [center_x, center_y - triangle_size],
                    [center_x - triangle_size, center_y + triangle_size],
                    [center_x + triangle_size, center_y + triangle_size]
                ], np.int32)
                cv2.fillPoly(display_frame, [pts], (0, 0, 255))
                cv2.polylines(display_frame, [pts], True, (255, 255, 255), 2)
                cv2.putText(display_frame, "!", (center_x - 8, center_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Mostrar frame
            window_name = "Sistema de Deteccao de Quedas - Simulacao"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, display_frame)
            
            frame_idx += 1
        
        # Controles
        key = cv2.waitKey(int(1000 / FPS_SIMULATION)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Pausado" if paused else "Retomado")
        elif key == ord('r'):
            frame_idx = 0
            frames_queue = []
            print("Reiniciado")
    
    cv2.destroyAllWindows()
    print("\nSimulacao finalizada!")

if __name__ == "__main__":
    simulate_fall_detection()
