# Plano para Fortalecer o Artigo para BRACIS

*Atualizado em 04/05/2026 — reflete o estado real do projeto após os experimentos de ablação.*

---

## Estado atual (resumo)

| Item | Status |
|---|---|
| Split por vídeo (GroupShuffleSplit) | ✅ Implementado |
| Baselines (CNN-only, frozen) | ✅ Implementado |
| Ablation study (6 experimentos) | ✅ Rodado e logado |
| Métricas reais coletadas | ✅ Todos os runs em `logs/` |
| Artigo atualizado com ablation + split correto | ✅ Feito em `docs/artigo.tex` |
| **Artigo em inglês** | ❌ Ainda em português |
| **Formato LLNCS (Springer)** | ❌ Ainda usa `article` class |
| **Citações numéricas (splncs04)** | ❌ Ainda usa natbib autor-ano |
| **Autores anonimizados (double-blind)** | ❌ Nomes expostos |
| Related Work com referências 2022-2026 | ❌ Para fazer |
| Medição de latência/FPS | ❌ Para fazer |

---

## Problema científico residual

O resultado de 100% de acurácia foi mantido mesmo com o split por vídeo — o que é positivo para a integridade das métricas, mas revela um **teto de desempenho do dataset**: o CNN-only (sem LSTM) atinge o mesmo resultado que a arquitetura completa. Um revisor vai questionar a contribuição da LSTM.

**Estratégia adotada:** reportar esse achado honestamente na discussão, apresentando como evidência de limitação do dataset (ambiente controlado, baixa variabilidade), não como falha do método. O argumento é que a validação em datasets mais complexos é trabalho futuro necessário.

---

## Bloqueadores obrigatórios antes da submissão

Estes quatro itens causam **rejeição de mesa** (sem revisão de conteúdo) se não corrigidos.

---

### Ação 1 — Traduzir o artigo para inglês

**Arquivo:** `docs/artigo.tex` → renomear para `docs/artigo_bracis.tex` após tradução

**Status:** ❌ Pendente

Todo o conteúdo deve ser traduzido, incluindo:
- Título, abstract, palavras-chave
- Todo o corpo do texto
- Legendas de figuras e tabelas
- Conteúdo de tabelas (rótulos de linhas/colunas)
- Nomes de seções e subseções

Termos técnicos em inglês já no texto (MobileNetV2, LSTM, fine-tuning, etc.) permanecem como estão.

---

### Ação 2 — Migrar para a classe LLNCS (Springer)

**Arquivo:** `docs/artigo_bracis.tex`

**Status:** ❌ Pendente

Substituir:
```latex
\documentclass[12pt,a4paper]{article}
```
por:
```latex
\documentclass{llncs}
```

Baixar o pacote LLNCS em: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

Ajustes necessários ao migrar:
- Remover pacotes incompatíveis (`geometry`, `authblk`, `fontenc` com T1)
- Substituir `\author[1]{...}\affil[1]{...}` pelo formato LLNCS: `\author{...}\institute{...}`
- Verificar que `\maketitle` e `\begin{abstract}` seguem o padrão LLNCS
- Remover `\usepackage[margin=2.5cm]{geometry}` — LLNCS tem margens próprias
- Checar que figuras e tabelas cabem no limite de página (~8–12 páginas)

---

### Ação 3 — Trocar sistema de citações para numérico

**Arquivo:** `docs/artigo_bracis.tex`

**Status:** ❌ Pendente

Remover:
```latex
\usepackage[round,authoryear]{natbib}
```

Substituir por:
```latex
\bibliographystyle{splncs04}
```

Ou, se usar `thebibliography` manual, mudar o estilo de `\bibitem[Autor(ano)]{chave}` para `\bibitem{chave}` com numeração sequencial, e trocar todos os `\citet{}` / `\citep{}` por `\cite{}`.

---

### Ação 4 — Anonimizar para double-blind

**Arquivo:** `docs/artigo_bracis.tex`

**Status:** ❌ Pendente

Remover ou substituir por `[Omitted for blind review]`:
- Nomes dos autores
- Afiliação institucional (UEA, Manaus)
- Agradecimentos ao PAIC/FAPEAM (citar financiamento sem identificar)
- URL do repositório GitHub (ou substituir por `[Repository omitted for blind review]`)

---

## Melhorias científicas (não são bloqueadores, mas aumentam chance de aceite)

### Ação 5 — Atualizar Related Work com referências 2022-2026

**Arquivo:** `docs/artigo_bracis.tex`, seção 2

**Status:** ❌ Pendente

Adicionar no mínimo 3-4 trabalhos recentes de fall detection:
- Abordagens com pose estimation (MediaPipe Pose / OpenPose) para detecção de quedas
- Transformers aplicados a vídeo (ex: Video Swin Transformer)
- Trabalhos de fall detection em edge devices (Raspberry Pi, Jetson Nano)
- Revisão recente do estado da arte (2022-2025)

Sugestão de estrutura de busca: Google Scholar com "fall detection deep learning 2023 2024 2025", filtrando por venues como IEEE JBHI, Sensors (MDPI), Pattern Recognition Letters.

---

### Ação 6 — Medir e reportar latência de inferência

**Arquivo:** `scripts/main.py` ou script dedicado; resultado vai para seção 4.5 do artigo

**Status:** ❌ Pendente

Coletar e reportar:
- FPS médio em inferência contínua (modo `--skip-frames 5`)
- Latência por predição em ms (tempo de `model(input)` isolado)
- Comparação modelo `.keras` vs `.tflite` (tamanho: 20,8 MB vs 2,4 MB; tempo de inferência)
- Hardware: Intel i7 8ª geração, 8 GB RAM, sem GPU

Exemplo de trecho para o artigo:
> "The system achieved an average inference latency of X ms per prediction on CPU (Intel Core i7-8th gen, 8 GB RAM), enabling near-real-time operation at Y FPS with 5-frame skip."

---

### Ação 7 — Fortalecer a justificativa da contribuição da LSTM

**Arquivo:** `docs/artigo_bracis.tex`, seções 2 e 5

**Status:** ❌ Pendente

O CNN-only atingiu 100% no URFD. O revisor vai perguntar: *"why use LSTM?"*. Opções para responder isso:

**Opção A (argumentação teórica):** Argumentar que o URFD é um dataset de teto fácil, e citar trabalhos que mostram que a modelagem temporal é discriminativa em datasets mais desafiadores (quedas em ambientes não controlados, atividades ambíguas como sentar rapidamente). A LSTM é necessária para generalização, não para este benchmark específico.

**Opção B (experimento adicional — ideal):** Rodar o baseline CNN-only em um segundo dataset (ex: Le2i Fall Detection, disponível publicamente) e mostrar que, nesse dataset mais difícil, o CNN+LSTM supera o CNN-only. Isso validaria empiricamente a contribuição da LSTM.

Recomendação: implementar a Opção A no texto e tentar a Opção B se houver tempo.

---

### Ação 8 (bônus) — Teste qualitativo com webcam

**Status:** ❌ Pendente

Rodar `scripts/main.py` em ambiente diferente do dataset (própria casa, iluminação variada) e reportar resultados qualitativos:
- Quantos falsos positivos por minuto em uso normal
- Se o mecanismo de confirmação temporal (θ=0,75, N=3) é suficiente
- Tipos de atividade que geram falsos positivos

Adicionar como parágrafo curto na seção de Resultados. Revisores BRACIS valorizam evidência de funcionamento fora do dataset.

---

## Ordem de execução recomendada

```
1. ❌ Ação 1 — Traduzir (pré-requisito para tudo)
2. ❌ Ação 2 — Migrar para LLNCS (junto com a tradução)
3. ❌ Ação 3 — Trocar citações (junto com Ação 2)
4. ❌ Ação 4 — Anonimizar (último passo antes de submeter)
5. ❌ Ação 5 — Related Work atualizado (pode fazer em paralelo com Ações 1-3)
6. ❌ Ação 6 — Medir latência (30 min de trabalho, alto impacto)
7. ❌ Ação 7 — Justificativa LSTM (mínimo: Opção A no texto)
8. ❌ Ação 8 — Webcam (bônus, se houver tempo)
```

---

## Critérios de prontidão para submissão

O artigo está pronto para submeter quando:

- [ ] Texto 100% em inglês
- [ ] Compila sem erros com `\documentclass{llncs}`
- [ ] Todas as citações são numéricas `[1]`, `[2]`...
- [ ] Nenhum nome de autor ou instituição visível no PDF
- [ ] Número de páginas dentro do limite do BRACIS (verificar no CFP)
- [ ] Pelo menos 3 referências de 2022-2026 na seção Related Work
- [ ] Latência de inferência reportada com valores concretos
- [ ] Contribuição da LSTM justificada mesmo com CNN-only igualando em URFD
