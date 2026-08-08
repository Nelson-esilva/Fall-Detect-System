"""
Preprocessa o Le2i Fall Detection Dataset (Charfi et al., 2013) gerando
clipes .avi de Normal e Fall em data/le2i/, prontos para uso com train_model.py.

Estrutura esperada do dataset baixado:
  <DATASET_DIR>/
    Coffee_room_01/Coffee_room_01/
      Annotation_files/video (N).txt
      Videos/video (N).avi
    Coffee_room_02/...
    Home_01/...
    Home_02/...
    Lecture_room/...   <- sem anotações, ignorado
    Office/...         <- sem anotações, ignorado

Formato do arquivo de anotação:
  linha 1: frame_inicio_queda
  linha 2: frame_fim_queda
  linhas 3+: frame,pose,x1,y1,x2,y2  (bounding box por frame)

Saída:
  data/le2i/Normal/<env>_video<N>_normal.avi
  data/le2i/Fall/<env>_video<N>_fall.avi

Uso:
  python scripts/prepare_le2i.py
  python scripts/prepare_le2i.py --dataset-dir "C:/caminho/para/le2i"
  python scripts/prepare_le2i.py --img-size 128 --pre-fall-margin 5
"""

import argparse
import os
import sys
import re
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH

# Caminho padrão onde o kagglehub salva o dataset
DEFAULT_DATASET_DIR = (
    Path.home()
    / ".cache/kagglehub/datasets/tuyenldvn/falldataset-imvia/versions/2"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "le2i"

# Ambientes com anotações (subpasta interna repete o nome)
ANNOTATED_ENVS = [
    ("Coffee_room_01", "Coffee_room_01"),
    ("Coffee_room_02", "Coffee_room_02"),
    ("Home_01",        "Home_01"),
    ("Home_02",        "Home_02"),
]


# ---------------------------------------------------------------------------
# Parsing de anotação
# ---------------------------------------------------------------------------

def parse_annotation(ann_path: Path):
    """
    Retorna (fall_start, fall_end) com base 0 (índice de frame).
    O arquivo usa base 1; convertemos subtraindo 1.

    Casos especiais:
    - Se a primeira linha não é um inteiro puro (ex: "1,1,72,58,132,170"),
      o arquivo é uma anotação ADL sem queda → retorna (-1, -1).
    - Se fall_start == fall_end == 0, o dataset indica ADL sem queda → (-1, -1).
    - Retorna None apenas em caso de erro de leitura irrecuperável.
    """
    try:
        lines = ann_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = [l.strip() for l in lines if l.strip()]

        # Verifica se a primeira linha é um inteiro puro
        if "," in lines[0]:
            # Formato alternativo: arquivo começa direto com bounding boxes → ADL puro
            return (-1, -1)

        fall_start_raw = int(lines[0])
        fall_end_raw   = int(lines[1])

        # Convenção: 0,0 significa ADL sem queda
        if fall_start_raw == 0 and fall_end_raw == 0:
            return (-1, -1)

        return fall_start_raw - 1, fall_end_raw - 1  # converter para base 0
    except Exception as e:
        print(f"  AVISO: falha ao ler {ann_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Leitura de vídeo
# ---------------------------------------------------------------------------

def read_frames(video_path: Path, img_size: int):
    """Lê todos os frames de um vídeo e redimensiona para img_size x img_size."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (img_size, img_size)))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Escrita de clipe
# ---------------------------------------------------------------------------

def write_clip(frames: list, output_path: Path, fps: float = 25.0):
    """Escreve uma lista de frames como arquivo .avi."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return True


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------

def process_environment(env_outer: str, env_inner: str, dataset_dir: Path,
                         output_dir: Path, img_size: int, pre_fall_margin: int):
    """
    Processa todos os vídeos de um ambiente e salva clipes Normal/Fall.
    Retorna dicionário com contagens.
    """
    env_dir  = dataset_dir / env_outer / env_inner
    ann_dir  = env_dir / "Annotation_files"
    vid_dir  = env_dir / "Videos"

    if not ann_dir.exists() or not vid_dir.exists():
        print(f"  Pulando {env_outer}: pastas Annotation_files ou Videos não encontradas.")
        return {"skipped": 0, "normal": 0, "fall": 0, "errors": 0}

    env_tag = env_outer.lower().replace(" ", "_")
    normal_dir = output_dir / "Normal"
    fall_dir   = output_dir / "Fall"
    normal_dir.mkdir(parents=True, exist_ok=True)
    fall_dir.mkdir(parents=True, exist_ok=True)

    # Coletar pares (vídeo, anotação) pelo índice extraído do nome
    video_files = sorted(vid_dir.glob("*.avi"), key=_video_sort_key)

    counts = {"normal": 0, "fall": 0, "skipped": 0, "errors": 0}

    for video_path in video_files:
        idx = _extract_index(video_path.stem)
        if idx is None:
            counts["errors"] += 1
            continue

        ann_path = ann_dir / f"video ({idx}).txt"
        if not ann_path.exists():
            print(f"  AVISO: anotação não encontrada para {video_path.name}")
            counts["errors"] += 1
            continue

        result = parse_annotation(ann_path)
        if result is None:
            counts["errors"] += 1
            continue
        fall_start, fall_end = result

        frames = read_frames(video_path, img_size)
        n_frames = len(frames)

        if n_frames == 0:
            print(f"  AVISO: vídeo vazio {video_path.name}")
            counts["errors"] += 1
            continue

        tag = f"{env_tag}_video{idx}"

        # Vídeo ADL puro (sem queda): salvar tudo como Normal
        if fall_start == -1:
            if len(frames) >= SEQUENCE_LENGTH:
                out_path = normal_dir / f"{tag}_normal.avi"
                if write_clip(frames, out_path):
                    counts["normal"] += 1
                    print(f"  Normal  | {tag} | {len(frames)} frames (ADL puro) -> {out_path.name}")
                else:
                    counts["errors"] += 1
            else:
                print(f"  Normal  | {tag} | {len(frames)} frames insuficientes "
                      f"(mínimo {SEQUENCE_LENGTH}), ignorado.")
                counts["skipped"] += 1
            continue

        # Clipe Normal: frames ANTES da queda (com margem de segurança)
        normal_end = max(0, fall_start - pre_fall_margin)
        normal_frames = frames[:normal_end]

        # Clipe Fall: frames A PARTIR do início da queda até o fim
        fall_frames = frames[fall_start:]

        # Salvar Normal (só se tiver frames suficientes para ao menos uma janela)
        if len(normal_frames) >= SEQUENCE_LENGTH:
            out_path = normal_dir / f"{tag}_normal.avi"
            if write_clip(normal_frames, out_path):
                counts["normal"] += 1
                print(f"  Normal  | {tag} | {len(normal_frames)} frames -> {out_path.name}")
            else:
                counts["errors"] += 1
        else:
            print(f"  Normal  | {tag} | {len(normal_frames)} frames insuficientes "
                  f"(mínimo {SEQUENCE_LENGTH}), ignorado.")
            counts["skipped"] += 1

        # Salvar Fall (só se tiver frames suficientes)
        if len(fall_frames) >= SEQUENCE_LENGTH:
            out_path = fall_dir / f"{tag}_fall.avi"
            if write_clip(fall_frames, out_path):
                counts["fall"] += 1
                print(f"  Fall    | {tag} | {len(fall_frames)} frames -> {out_path.name}")
            else:
                counts["errors"] += 1
        else:
            print(f"  Fall    | {tag} | {len(fall_frames)} frames insuficientes "
                  f"(mínimo {SEQUENCE_LENGTH}), ignorado.")
            counts["skipped"] += 1

    return counts


def _extract_index(stem: str):
    """Extrai o número de 'video (N)' -> N."""
    m = re.search(r"\((\d+)\)", stem)
    return int(m.group(1)) if m else None


def _video_sort_key(p: Path):
    idx = _extract_index(p.stem)
    return idx if idx is not None else 9999


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocessa o Le2i Fall Detection Dataset.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
        help=f"Caminho para a pasta raiz do Le2i (default: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Pasta de saída (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--img-size", type=int, default=IMG_HEIGHT,
        help=f"Tamanho de redimensionamento dos frames (default: {IMG_HEIGHT})"
    )
    parser.add_argument(
        "--pre-fall-margin", type=int, default=5,
        help="Frames a ignorar antes do início da queda no clipe Normal (default: 5)"
    )
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"ERRO: dataset não encontrado em {args.dataset_dir}")
        print("Passe o caminho correto com --dataset-dir ou verifique o download.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("PREPARAÇÃO DO Le2i FALL DETECTION DATASET")
    print(f"{'='*60}")
    print(f"  Dataset : {args.dataset_dir}")
    print(f"  Saída   : {args.output_dir}")
    print(f"  Img size: {args.img_size}x{args.img_size}")
    print(f"  Margem  : {args.pre_fall_margin} frames antes da queda ignorados no clipe Normal")
    print(f"  Janela  : mínimo {SEQUENCE_LENGTH} frames por clipe")
    print()

    total = {"normal": 0, "fall": 0, "skipped": 0, "errors": 0}

    for env_outer, env_inner in ANNOTATED_ENVS:
        print(f"\n[{env_outer}]")
        counts = process_environment(
            env_outer, env_inner,
            args.dataset_dir, args.output_dir,
            args.img_size, args.pre_fall_margin,
        )
        for k in total:
            total[k] += counts[k]

    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")
    print(f"  Clipes Normal salvos : {total['normal']}")
    print(f"  Clipes Fall salvos   : {total['fall']}")
    print(f"  Clipes curtos (skip) : {total['skipped']}")
    print(f"  Erros                : {total['errors']}")
    print(f"\n  Saída: {args.output_dir}/Normal/  e  {args.output_dir}/Fall/")
    print()
    print("Próximo passo:")
    print(f"  python scripts/train_model.py --data-dir {args.output_dir} --tag le2i-cnn-lstm")
    print()


if __name__ == "__main__":
    main()
