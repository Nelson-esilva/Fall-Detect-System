import os
import cv2
import glob
import shutil

# Configurações
INPUT_DIR = 'UR_Fall_Downloads'
OUTPUT_DIR = 'data/raw'
IMG_HEIGHT, IMG_WIDTH = 224, 224

def find_images_in_dir(start_dir):
    """
    Procura recursivamente pela pasta que contém arquivos .png
    """
    # Primeiro, verifica se a pasta atual já tem pngs
    pngs = glob.glob(os.path.join(start_dir, "*.png"))
    if pngs:
        return start_dir, sorted(pngs)
    
    # Se não, procura nas subpastas
    for root, dirs, files in os.walk(start_dir):
        pngs = glob.glob(os.path.join(root, "*.png"))
        if pngs:
            return root, sorted(pngs)
            
    return None, None

def create_video(image_paths, output_path):
    if not image_paths:
        return

    # Ler primeira imagem para garantir dimensões (embora vamos redimensionar)
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"ERRO: Não foi possível ler {image_paths[0]}")
        return

    # Criar VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (IMG_WIDTH, IMG_HEIGHT))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            out.write(resized)
        else:
            print(f"Aviso: Falha ao ler imagem {img_path}")

    out.release()
    print(f"Vídeo criado: {output_path} ({len(image_paths)} frames)")

def process_extracted_folders():
    # Criar pastas de saída
    os.makedirs(os.path.join(OUTPUT_DIR, 'Fall'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Normal'), exist_ok=True)

    # Listar pastas dentro de UR_Fall_Downloads
    if not os.path.exists(INPUT_DIR):
        print(f"Pasta {INPUT_DIR} não encontrada!")
        return

    items = os.listdir(INPUT_DIR)
    folders = [os.path.join(INPUT_DIR, item) for item in items if os.path.isdir(os.path.join(INPUT_DIR, item))]
    
    if not folders:
        print(f"Nenhuma pasta encontrada em {INPUT_DIR}.")
        return

    print(f"Encontradas {len(folders)} pastas. Processando...")

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        
        # Ignorar pastas que não sejam do dataset (ex: __pycache__)
        if not ("fall-" in folder_name.lower() or "adl-" in folder_name.lower()):
            continue

        # Identificar categoria
        if "fall" in folder_name.lower():
            category = "Fall"
        elif "adl" in folder_name.lower():
            category = "Normal"
        else:
            continue
            
        # Encontrar onde estão as imagens realmente
        img_dir, images = find_images_in_dir(folder_path)
        
        if not images:
            print(f"AVISO: Nenhuma imagem encontrada em {folder_name}")
            continue
            
        # Nome do vídeo de saída
        video_name = f"{folder_name}.avi"
        output_path = os.path.join(OUTPUT_DIR, category, video_name)
        
        create_video(images, output_path)

    print("\nProcessamento concluído! Agora rode 'train_model.py'.")

if __name__ == "__main__":
    process_extracted_folders()
