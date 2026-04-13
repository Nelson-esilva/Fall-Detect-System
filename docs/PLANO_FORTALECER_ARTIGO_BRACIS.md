# Plano para Fortalecer o Artigo para BRACIS

## Problema central

O split atual em `train_model.py` usa `train_test_split` por **janela** (stride 10 em vídeos de ~100 frames), permitindo que janelas do mesmo vídeo caiam em treino e teste. Isso invalida as métricas de 100% e é o ponto mais crítico que um revisor apontaria.

Além disso, faltam baselines, ablation study e referências recentes.

---

## Ação 1 — Split por vídeo (corrigir data leakage)

**Arquivo:** `scripts/train_model.py`

**Status:** ✅ Concluído

Atualmente `build_dataset()` retorna apenas `(X, y)` sem rastrear a origem de cada amostra. O fix:

- Modificar `build_dataset()` para retornar também um array `video_ids` (índice do vídeo de origem para cada janela)
- Substituir `train_test_split(X, y, ...)` por um split baseado em `GroupShuffleSplit` do scikit-learn, onde o grupo é o `video_id`
- Isso garante que TODAS as janelas de um vídeo ficam em treino OU em teste, nunca em ambos

Lógica concreta:
```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=video_ids))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

**Impacto esperado:** a acurácia provavelmente vai cair para 85-95%, mas será uma métrica real e confiável.

---

## Ação 2 — Baselines (mínimo 2 modelos de comparação)

**Arquivo:** `src/model.py` (novas funções) + `scripts/train_model.py` (flag de seleção)

**Status:** ✅ Concluído

Implementar pelo menos:

- **Baseline 1 — CNN-only (sem LSTM):** MobileNetV2 + GlobalAveragePooling + Dense. Recebe frames individuais, não sequências. Mostra que a LSTM agrega valor temporal.
- **Baseline 2 — CNN-LSTM frozen (sem fine-tuning):** O modelo atual com `fine_tune_layers=0`. Mostra que o fine-tuning agrega valor.
- **Baseline 3 (opcional) — Threshold simples:** Calcula diferença de frames consecutivos (motion energy) e classifica por limiar. Mostra que deep learning supera métodos clássicos.

Cada baseline deve ser treinado com o mesmo split por vídeo e gerar seus próprios logs.

---

## Ação 3 — Ablation Study

**Arquivo:** `scripts/train_model.py` (parametrização via argparse ou config)

**Status:** ✅ Concluído

Rodar experimentos variando um componente por vez:

| Experimento | O que muda |
|---|---|
| Full model | CNN-LSTM, fine-tune 30, augmentation, 128x128 |
| Sem fine-tuning | `fine_tune_layers=0` |
| Sem augmentation | Desativar flip+brilho |
| Resolução 224 | `IMG_HEIGHT=IMG_WIDTH=224` |
| LSTM 128 unidades | Alterar unidades da LSTM |

O resultado é uma tabela no artigo mostrando o impacto de cada componente.

---

## Ação 4 — Retreinar e coletar métricas reais

**Status:** ⏳ Pendente

Rodar todos os experimentos (modelo principal + baselines + ablations) com o split por vídeo e salvar os logs em `logs/run-*`. As métricas reais substituirão os 100% no artigo.

Script de batch: `scripts/run_experiments.py`

---

## Ação 5 — Atualizar o artigo (texto)

**Arquivo:** `docs/artigo_bracis.tex`

**Status:** ⏳ Pendente

Mudanças no texto:
- **Related Work:** adicionar 3-5 referências de 2022-2026 (Vision Transformers para vídeo, MediaPipe/pose estimation, trabalhos recentes de fall detection com LSTM/Transformer)
- **Methodology:** descrever o split por vídeo e justificar
- **Results:** substituir a tabela de métricas com os valores reais do split por vídeo; adicionar tabela de baselines e tabela de ablation study
- **Discussion:** remover a ressalva sobre data leakage (já corrigido); discutir o gap entre split por janela e por vídeo; comparar baselines
- **Comparison table:** adicionar os baselines na tabela de comparação

---

## Ação 6 (bonus) — Teste qualitativo com webcam

**Status:** ⏳ Pendente

Se possível, rodar `scripts/main.py` com a webcam em um ambiente diferente do dataset e reportar resultados qualitativos (funciona/não funciona, tipos de falsos positivos). Isso adiciona uma seção de "real-world qualitative evaluation" que revisores valorizam.

---

## Ordem de execução

1. ✅ Ação 1 (split por vídeo) — pré-requisito para tudo
2. ✅ Ação 2 (baselines) — código dos modelos alternativos
3. ✅ Ação 3 (ablation) — parametrização dos experimentos
4. ⏳ Ação 4 (retreinar) — rodar tudo e coletar métricas
5. ⏳ Ação 5 (atualizar artigo) — com as métricas reais
6. ⏳ Ação 6 (teste webcam) — se houver tempo
