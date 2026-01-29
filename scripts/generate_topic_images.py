import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _save(fig, filename: str) -> str:
    out_path = os.path.join(ROOT_DIR, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _box(ax, x, y, w, h, title, lines=None, fc="#F7F7FB", ec="#2D2A32"):
    rect = Rectangle((x, y), w, h, linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=11, fontweight="bold")
    if lines:
        ax.text(x + 0.03, y + h - 0.13, "\n".join(lines), ha="left", va="top", fontsize=9)
    return rect


def _arrow(ax, x1, y1, x2, y2, color="#2D2A32"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=color)
    ax.add_patch(arr)
    return arr


def make_otimizacao_modelo():
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.97,
        "Otimização do Modelo (CNN + LSTM)\nAjuste de hiperparâmetros para melhor generalização em ângulos de câmera",
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    # Pipeline boxes
    _box(
        ax,
        0.05,
        0.60,
        0.36,
        0.24,
        "Busca de Hiperparâmetros",
        [
            "• learning rate: 1e-4, 3e-4, 1e-3",
            "• dropout: 0.3, 0.5, 0.6",
            "• early stopping (val_loss)",
            "• checkpoint (val_accuracy)",
        ],
    )
    _box(
        ax,
        0.46,
        0.60,
        0.24,
        0.24,
        "Avaliação",
        [
            "• acurácia/val_loss",
            "• matriz de confusão",
            "• precisão e recall",
        ],
        fc="#F2FBF8",
    )
    _box(
        ax,
        0.73,
        0.60,
        0.22,
        0.24,
        "Seleção Final",
        [
            "• melhor trade-off",
            "  entre generalização",
            "  e estabilidade",
        ],
        fc="#FFF7F0",
    )

    _arrow(ax, 0.41, 0.72, 0.46, 0.72)
    _arrow(ax, 0.70, 0.72, 0.73, 0.72)

    # "Ângulos de câmera" mini-cards
    _box(ax, 0.05, 0.18, 0.27, 0.30, "Ângulos de Câmera (teste)", ["• frontal", "• lateral", "• superior", "• diagonal"])
    _box(
        ax,
        0.35,
        0.18,
        0.27,
        0.30,
        "Indicadores de Robustez",
        ["• variação de iluminação", "• oclusões parciais", "• mudança de cenário"],
        fc="#F0F6FF",
    )
    _box(
        ax,
        0.65,
        0.18,
        0.30,
        0.30,
        "Resultado Esperado",
        ["• menos falsos positivos", "• detecção consistente", "• melhor generalização"],
        fc="#F7F7FB",
    )
    _arrow(ax, 0.32, 0.33, 0.35, 0.33)
    _arrow(ax, 0.62, 0.33, 0.65, 0.33)

    # Mini chart (stylized) for LR vs dropout grid
    ax.text(0.05, 0.53, "Mapa de validação (ilustrativo):", fontsize=10, fontweight="bold")
    x0, y0, cell = 0.30, 0.52, 0.035
    for i in range(3):
        for j in range(3):
            # color gradient: pretend better in center
            d = abs(i - 1) + abs(j - 1)
            fc = ["#2A9D8F", "#E9C46A", "#E76F51"][min(d, 2)]
            ax.add_patch(Rectangle((x0 + j * cell, y0 - i * cell), cell, cell, facecolor=fc, edgecolor="#2D2A32", linewidth=0.6))
    ax.text(x0 + 0.11, y0 + 0.01, "dropout →", fontsize=8)
    ax.text(x0 - 0.06, y0 - 0.02, "lr\n↓", fontsize=8, ha="center")
    ax.text(x0 + 0.002, y0 + 0.04, "verde=melhor", fontsize=8)

    return _save(fig, "otimizacao_modelo.png")


def make_protocolos_alerta_fcm():
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.97,
        "Pesquisa de Protocolos de Alerta\nArquitetura de notificações em tempo real com Firebase Cloud Messaging (FCM)",
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    # Boxes
    _box(
        ax,
        0.05,
        0.62,
        0.28,
        0.22,
        "Detecção (Edge/PC)",
        [
            "• câmera (OpenCV)",
            "• buffer 20 frames",
            "• inferência CNN+LSTM",
            "• evento: QUEDA",
        ],
        fc="#F2FBF8",
    )
    _box(
        ax,
        0.37,
        0.62,
        0.28,
        0.22,
        "Backend / API",
        [
            "• validação do evento",
            "• registro (timestamp)",
            "• regras de reenvio",
        ],
        fc="#F0F6FF",
    )
    _box(
        ax,
        0.69,
        0.62,
        0.26,
        0.22,
        "Firebase (FCM)",
        [
            "• token do device",
            "• envio de push",
            "• prioridade alta",
        ],
        fc="#FFF7F0",
    )

    _arrow(ax, 0.33, 0.73, 0.37, 0.73)
    _arrow(ax, 0.65, 0.73, 0.69, 0.73)

    _box(
        ax,
        0.15,
        0.20,
        0.34,
        0.30,
        "App Mobile (Flutter/React Native)",
        [
            "• recebe notificação",
            "• abre tela de alerta",
            "• confirma/aciona cuidador",
            "• histórico de ocorrências",
        ],
    )
    _box(
        ax,
        0.55,
        0.20,
        0.40,
        0.30,
        "Ações/Resposta",
        [
            "• som local (ESP32/RPi)",
            "• ligação/SMS (opcional)",
            "• escalonamento por tempo",
            "• registro para auditoria",
        ],
        fc="#F7F7FB",
    )

    # Down arrows from FCM to Mobile and actions
    _arrow(ax, 0.82, 0.62, 0.32, 0.50)  # FCM -> Mobile
    _arrow(ax, 0.82, 0.62, 0.75, 0.50)  # FCM -> Actions

    ax.text(
        0.05,
        0.10,
        "Observação: o FCM pode ser acionado por backend (recomendado) para autenticação, logs e controle de reenvio.",
        fontsize=9,
        color="#333333",
    )

    return _save(fig, "protocolos_alerta_fcm.png")


def make_documentacao_tecnica():
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.97,
        "Documentação Técnica\nOrganização de scripts, logs de performance e versionamento do projeto",
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    # Left: folder structure
    _box(
        ax,
        0.05,
        0.18,
        0.42,
        0.70,
        "Estrutura sugerida",
        [
            "Fall-Detect-System/",
            "  src/model.py",
            "  main.py",
            "  train_model.py",
            "  prepare_ur_fall.py",
            "  models/  (artefatos .h5)",
            "  logs/    (csv/json, plots)",
            "  docs/    (relatórios)",
            "  configs/ (yaml/json)",
        ],
        fc="#F7F7FB",
    )

    # Right: versioning & logs
    _box(
        ax,
        0.52,
        0.60,
        0.43,
        0.28,
        "Versionamento (Git)",
        [
            "• commits pequenos e frequentes",
            "• tags de versão (v0.x)",
            "• ignore: datasets grandes",
            "• rastreio de configs",
        ],
        fc="#F0F6FF",
    )
    _box(
        ax,
        0.52,
        0.28,
        0.43,
        0.28,
        "Logs de Treino",
        [
            "• loss/accuracy por época",
            "• matriz de confusão",
            "• hiperparâmetros (lr/dropout)",
            "• data/hora + dataset usado",
        ],
        fc="#F2FBF8",
    )

    _arrow(ax, 0.47, 0.72, 0.52, 0.74)
    _arrow(ax, 0.47, 0.40, 0.52, 0.42)

    # Mini legend cards
    _box(
        ax,
        0.52,
        0.16,
        0.21,
        0.09,
        "Saídas",
        ["• modelos\n• gráficos\n• relatórios"],
        fc="#FFF7F0",
    )
    _box(
        ax,
        0.74,
        0.16,
        0.21,
        0.09,
        "Reprodutibilidade",
        ["• seed\n• configs\n• ambiente"],
        fc="#FFF7F0",
    )

    return _save(fig, "documentacao_tecnica.png")


def main():
    paths = [
        make_otimizacao_modelo(),
        make_protocolos_alerta_fcm(),
        make_documentacao_tecnica(),
    ]
    print("Imagens geradas:")
    for p in paths:
        print("-", p)


if __name__ == "__main__":
    main()

