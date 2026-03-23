# Histórico da Sessão — 22 de Março de 2026

Registro completo das atividades realizadas nesta sessão de desenvolvimento, cobrindo limpeza do repositório, documentação para GitHub e otimizações de performance.

---

## 1. Auditoria e Limpeza do Repositório

### Arquivos analisados

Todos os arquivos do projeto foram auditados para identificar redundâncias, órfãos e inconsistências.

### Problemas encontrados e corrigidos

| Problema | Ação |
|---|---|
| `.vscode/settings.json` estava sendo rastreado pelo Git, mas `.gitignore` já ignorava `.vscode/` | Removido do tracking com `git rm --cached` |
| `docs/ESTRUTURA_PROJETO.md` não mencionava a pasta `mobile/` | Atualizado com seção completa do mobile |
| `docs/ESP32_INTEGRATION.md` referenciava `main_with_esp32.py` sem o prefixo `scripts/` | Corrigidos 3 caminhos para `scripts/main_with_esp32.py` |
| `docs/ESP32_INTEGRATION.md` passo 3 apontava para o script em vez de `configs/config.py` | Corrigido |
| README.md desatualizado (não incluía mobile, diagrama de arquitetura, etc.) | Reescrito por completo |

### Conclusão da auditoria

Nenhum arquivo desnecessário foi encontrado além do `.vscode/settings.json`. O repositório estava limpo — todos os arquivos têm propósito claro.

---

## 2. Documentação Completa para GitHub

### README.md (reescrito)

O README principal foi reescrito para cobrir o sistema completo:

- **Diagrama ASCII** da arquitetura (PC → Mosquitto → App Mobile + ESP32)
- **Explicação do modelo CNN-LSTM** com as 4 camadas (MobileNetV2, TimeDistributed, LSTM, Dense)
- **Árvore de diretórios completa** incluindo `mobile/` com detalhamento de subpastas
- **Instruções de setup** para as 3 partes: detecção Python, ESP32, app mobile
- **Exemplo de teste integrado** com `mosquitto_pub`
- **Tabela de stack tecnológico** com versões
- **Documentação do protocolo MQTT** (payload JSON, portas, transporte)
- **Links para todos os documentos** do repositório

### docs/ESTRUTURA_PROJETO.md (atualizado)

- Adicionada seção completa da pasta `mobile/` com árvore de arquivos, funcionalidades e como executar
- Removida referência a PDFs inexistentes
- Incluído `Relatorio-Mes4-PAIC.md` na listagem
- Adicionados links cruzados para documentação mobile
- Data atualizada para março 2026

### docs/ESP32_INTEGRATION.md (corrigido)

- Caminhos de scripts corrigidos (3 ocorrências)
- Passo 3 agora aponta para `configs/config.py`
- Adicionada seção sobre integração com app mobile via MQTT/WebSocket

---

## 3. Discussão sobre Reorganização em 3 Pastas

O usuário perguntou sobre reorganizar o projeto em 3 vertentes: `DetectModel/`, `Mobile/`, `Hardware/`.

### Decisão: não reorganizar

Motivos:
- O `mobile/` já está isolado como projeto independente
- O `hardware/` é um único arquivo `.ino`
- A estrutura plana (src/, scripts/, configs/) é padrão em projetos ML Python
- Todos os scripts Python usam `PROJECT_ROOT = Path(__file__).parent.parent` — mover para subpasta quebraria todos os imports
- O `.gitignore` e toda a documentação recém-atualizada precisariam ser reescritos
- Custo alto, benefício baixo

A separação visual já existe no README com seções claras para cada vertente.

---

## 4. Análise de Performance e Sugestões

### Contexto

O PC de treino (RX 7800 + Ryzen 5500) queimou. O notebook atual tem:
- 8 GB RAM
- MX110 (GPU fraca)
- Intel i7 8ª geração

### Problemas identificados no código original

1. **`model.predict()`** criava grafo de execução a cada chamada (~10x mais lento que `model()`)
2. **`list.pop(0)`** no buffer de frames era O(n)
3. **Predição a cada frame** (30x/segundo) — desnecessário
4. **Resolução 224x224** — alta demais para tarefa binária
5. **Sem TFLite** — modelo Keras pesado rodando em CPU pura
6. **Sem data augmentation** — dataset pequeno sem técnicas de aumento
7. **Pesos totalmente congelados** — sem fine-tuning para o domínio

---

## 5. Otimizações Implementadas

### 5.1 configs/config.py

Novos parâmetros adicionados:

```python
# Resolução reduzida (era 224x224)
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Inferência
INFERENCE_SKIP_FRAMES = 5       # predizer a cada 5 frames
CONFIDENCE_THRESHOLD = 0.5
USE_TFLITE = True               # usar TFLite quando disponível

# Treinamento
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
FINE_TUNE_LAYERS = 30           # descongelar últimas 30 camadas da MobileNetV2
USE_MIXED_PRECISION = False     # float16 para GPUs compatíveis

# TFLite
TFLITE_MODEL_PATH = MODELS_DIR / 'fall_model_cnn_lstm.tflite'
```

### 5.2 src/model.py

- `build_cnn_lstm_model()` agora aceita `img_height`, `img_width`, `fine_tune_layers` e `learning_rate` como parâmetros
- Fine-tuning configurável: descongelar as últimas N camadas da MobileNetV2 para o modelo aprender features específicas de quedas
- Nova função `convert_to_tflite()` com quantização INT8 opcional

### 5.3 scripts/main.py e scripts/main_with_esp32.py

| Antes | Depois |
|---|---|
| `model.predict(input_data, verbose=0)` | `model(input_data, training=False)` |
| `list` + `pop(0)` | `collections.deque(maxlen=20)` |
| Predição a cada frame | Predição a cada `INFERENCE_SKIP_FRAMES` frames |
| Só modelo Keras (.h5) | TFLite prioritário, fallback para Keras |
| Resolução fixa 224x224 | Configurável via `config.py` |

Ambos os scripts agora carregam TFLite automaticamente quando o arquivo `.tflite` existe, com fallback transparente para o modelo Keras.

### 5.4 scripts/train_model.py

| Antes | Depois |
|---|---|
| Todos os frames em memória como lista Python | `tf.data.Dataset` com prefetch |
| Sem augmentation | Flip horizontal + variação de brilho |
| Pesos 100% congelados | Fine-tuning das últimas 30 camadas |
| Só EarlyStopping | + `ReduceLROnPlateau` (fator 0.5, paciência 3) |
| Só exporta `.h5` | Exporta `.h5` + `.tflite` automaticamente |
| PosixPath não serializável no JSON | Convertido para `str()` |
| Paciência do EarlyStopping: 5 | Aumentada para 7 |

### Impacto esperado

| Mudança | Qualidade | Velocidade |
|---|---|---|
| Resolução 128x128 | -1 a 3% acurácia | ~3x mais rápido |
| `model()` em vez de `predict()` | sem impacto | ~2-10x mais rápido |
| Skip 5 frames | sem impacto | ~5x menos inferências |
| TFLite INT8 | -0.1 a 0.5% | ~2-4x mais rápido em CPU |
| Fine-tuning 30 camadas | **+3 a 8%** | treino mais lento |
| Data augmentation | **+2 a 5%** | treino mais lento |

**Balanço final:** o modelo otimizado tende a ser melhor que o original (fine-tuning + augmentation compensam a resolução menor) e a inferência fica dramaticamente mais rápida.

---

## 6. Estado do Dataset

Ao final da sessão:
- `data/raw/Fall/`: 7 vídeos (fall-01 a fall-07)
- `data/raw/Normal/`: 3 vídeos (adl-01 a adl-03)
- Dataset completo do UR Fall tem 30 quedas + 40 ADLs
- Recomendação: baixar mais vídeos de https://fenix.ur.edu.pl/mkepski/ds/uf.html

---

## 7. Commits Realizados

### Commit 1: `ff519f2`
```
refactor: atualizar documentação completa do projeto para GitHub
```
- README.md reescrito
- ESTRUTURA_PROJETO.md atualizado com mobile/
- ESP32_INTEGRATION.md corrigido
- .vscode/settings.json removido do tracking

### Commit 2: `d985df4`
```
perf: otimizar pipeline de inferência e treinamento para hardware limitado
```
- configs/config.py — novos parâmetros
- src/model.py — fine-tuning + TFLite
- scripts/main.py — deque, model(), skip frames, TFLite
- scripts/main_with_esp32.py — mesmas otimizações
- scripts/train_model.py — tf.data, augmentation, ReduceLR, TFLite export

---

## 8. Pendências

- [ ] Baixar mais vídeos do UR Fall Dataset (ADLs especialmente)
- [ ] Treinar o modelo com os dados disponíveis
- [ ] Testar inferência em tempo real no notebook
- [ ] Etapa 4 do mobile (notificações background + polish visual)
- [ ] Teste integrado completo: Python → Mosquitto → App + ESP32
- [ ] Build de produção do app (APK/AAB)
