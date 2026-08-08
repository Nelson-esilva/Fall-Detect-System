# Contexto do projeto — BRACIS 2026 e fortalecimento científico

Este documento resume o estado do **Fall-Detect-System**, decisões desta conversa e o que falta fazer, para não repetir contexto em novos chats com o assistente.

---

## 1. O que é o projeto

Sistema de **detecção de quedas em tempo real** com:

- **Visão computacional + deep learning:** modelo híbrido **CNN (MobileNetV2) + LSTM** sobre janelas de 20 frames (128×128), classificação binária Normal vs Fall.
- **Pipeline de dados:** UR Fall Detection Dataset; vídeos em `data/raw/` (classes `Fall/` e `Normal/`); janela deslizante com stride ~10.
- **Inferência:** `scripts/main.py`, `scripts/simulate_fall.py`; otimizações (skip de frames, TFLite, buffer `deque`, confirmação temporal).
- **Integração:** ESP32 (buzzer/LEDs), broker **MQTT** (Mosquitto), app **React Native/Expo** com alarme e histórico.

Configurações centrais: `configs/config.py` (ex.: `FINE_TUNE_LAYERS=30`, `SEQUENCE_LENGTH=20`, `FALL_CONFIDENCE_THRESHOLD`, `FALL_CONSECUTIVE_FRAMES`).

---

## 2. Situação “anterior” e problema científico

### Split por janela (legacy)

O treinamento original em `scripts/train_model.py` usava `train_test_split` **por amostra de janela**, não por vídeo. Janelas do **mesmo vídeo** podiam cair em treino e teste → **data leakage** e métricas artificialmente altas (ex.: relatório antigo com **100%** de acurácia nos logs).

Isso **não invalida** o sistema em si, mas **invalida** usar essas métricas como prova científica forte em artigo de conferência.

### O que revisores criticariam

- Métricas “perfeitas” sem split por vídeo.
- Falta de **baselines** e **ablation study**.
- Referências recentes e texto alinhado a boas práticas de avaliação.

---

## 3. Plano de fortalecimento (objetivo do artigo BRACIS)

Objetivo: tornar o artigo **competitivo** para **BRACIS / Springer LNCS (LNAI)** — revisão duplo-cego, inglês, citações numéricas, figuras/tabelas no estilo LNCS.

Plano detalhado (não editar o arquivo de plano original do Cursor): cópia em projeto em  
[`docs/PLANO_FORTALECER_ARTIGO_BRACIS.md`](PLANO_FORTALECER_ARTIGO_BRACIS.md).

### Já implementado no código

| Item | Onde |
|------|------|
| Split por vídeo (`GroupShuffleSplit`, `groups=video_ids`) | `scripts/train_model.py` — padrão `--split video`; `--split window` mantém legacy |
| `build_dataset()` retorna `(X, y, video_ids)` | `scripts/train_model.py` |
| Baseline CNN-only (sem LSTM temporal explícito) | `src/model.py` — `build_cnn_only_model` |
| LSTM com unidades parametrizáveis | `src/model.py` — `build_cnn_lstm_model(..., lstm_units=...)` |
| Flags: modelo, fine-tune, augmentation, img-size, lstm-units, tag | `scripts/train_model.py` (argparse) |
| Batch de experimentos | `scripts/run_experiments.py` |

### Baseline 3 (motion energy / threshold)

Mencionado no plano como opcional — **não** está implementado como script separado; pode ser futuro reforço.

### Pendente (antes do artigo final)

1. **Rodar treinos** (`run_experiments.py` ou comandos equivalentes) e guardar `logs/run-*` com `final_metrics.json`, `classification_report.txt`, curvas, matriz de confusão.
2. **Atualizar** [`docs/artigo_bracis.tex`](artigo_bracis.tex): métricas reais, tabelas de baselines/ablation, Related Work recente, metodologia descrevendo split por vídeo, Discussion sem depender do 100% “por janela”.
3. **(Opcional)** teste qualitativo com webcam / ambiente real e parágrafo curto no artigo.

---

## 4. Artigo LaTeX BRACIS

- Arquivo principal: [`docs/artigo_bracis.tex`](artigo_bracis.tex).
- Classe: `llncs`; autores anonimizados para double-blind.
- Bibliografia: estilo numérico (`splncs04` comentado no fim; entradas em `thebibliography`).
- Imagens esperadas no Overleaf (nomes usados no `.tex`): `pipeline_dados_branco.png`, `arquitetura_modelo_cnn_lstm_branco.png`, `arquitetura_sistema_branco.png`, `telas_app_mobile_branco.png`, `confusion_matrix.png`, `curves.png`.

### Correção LaTeX (URLs)

Foi adicionado `\usepackage{url}` antes de `\UrlFont`/`\urlstyle` para evitar erro de compilação (comandos do pacote `url`).

---

## 5. Prazos e estratégia de submissão (conversa)

- **Registro / resumo:** prazo próximo (data da conversa: abril/2026); o **artigo completo** tem prazo posterior (ex.: até dia **20** do mês — confirmar no site oficial BRACIS).
- Estratégia acordada: no registro, **abstract** descrevendo metodologia (split por vídeo, baselines, ablation) **sem** soar como “correção de erro”; nos dias seguintes, **rodar experimentos** e **preencher métricas** no PDF final.
- Resumo em inglês foi refinado para: problema → arquitetura → avaliação rigorosa → baselines/ablation → sistema IoT/mobile → frase de resultados **compatível** com o plano de experimentos (ex.: modelo completo supera baselines), **sem** números inventados até os runs terminarem.

---

## 6. Tópicos BRACIS (até 6) — sugestão da conversa

Ordem de encaixe sugerida:

1. **Visão computacional**
2. **Redes neurais e aprendizado profundo**
3. **Aprendizado de máquina**
4. **Aplicações da IA em bioinformática e engenharia biomédica**
5. **Sistemas híbridos de IA**
6. **IA centrada no ser humano**

**Não** foi recomendado “previsão e análise de séries temporais” para este trabalho (associação mais forte a séries históricas/tabular do que a sequências de vídeo).

---

## 7. Git / branch

- Branch de trabalho das alterações de fortalecimento: `feature/fortalecer-artigo-bracis` (push para `origin` foi feito na conversa).
- `main` pode estar atrás dessa branch até merge.

---

## 8. Comandos úteis (referência)

Ativar venv (Windows PowerShell, exemplo):

```powershell
.\.venv\Scripts\Activate.ps1
```

Treino padrão (split por vídeo):

```powershell
python scripts/train_model.py --split video --model cnn_lstm
```

Experimentos em lote (demorado):

```powershell
python scripts/run_experiments.py
```

---

## 9. Preferências do usuário (Cursor)

- Respostas em **português**.
- Citações de código com formato `início:fim:caminho` quando relevante.
- Evitar editar arquivos markdown não solicitados; este arquivo foi pedido explicitamente.

---

## 10. Próximo passo lógico

1. Rodar experimentos com split por vídeo e consolidar métricas.  
2. Atualizar `artigo_bracis.tex` com números reais e tabelas.  
3. Revisar abstract final do PDF para alinhar 100% com resultados obtidos.  
4. (Opcional) teste qualitativo com câmera real.

---

*Gerado como memória de contexto da conversa sobre fortalecimento BRACIS, abril/2026.*
