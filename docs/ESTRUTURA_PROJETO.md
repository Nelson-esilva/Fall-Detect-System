# Estrutura do Projeto - Sistema de Detecção de Quedas

Este documento descreve a organização completa do projeto.

## 📁 Visão Geral da Estrutura

```
Fall-Detect-System/
│
├── 📂 src/                    # Código fonte principal (módulos reutilizáveis)
├── 📂 scripts/                # Scripts executáveis principais
├── 📂 hardware/               # Código para hardware (ESP32)
├── 📂 mobile/                 # Aplicativo mobile (React Native + Expo)
├── 📂 tests/                  # Testes e validações
├── 📂 configs/                # Configurações centralizadas
├── 📂 docs/                   # Documentação geral
├── 📂 assets/                 # Recursos visuais (imagens, diagramas)
├── 📂 data/                   # Dados de treinamento
├── 📂 models/                 # Modelos treinados
├── 📂 logs/                   # Logs de treinamento e performance
│
├── 📄 README.md               # Documentação principal
├── 📄 requirements.txt         # Dependências Python
└── 📄 .gitignore              # Arquivos ignorados pelo Git
```

## 📂 Detalhamento das Pastas

### `src/` - Código Fonte Principal
Módulos reutilizáveis e componentes core do sistema.

- **`model.py`**: Arquitetura CNN-LSTM do modelo de detecção
- **`esp32_interface.py`**: Interface de comunicação com ESP32 (Serial/MQTT)

**Uso:** Importado por scripts em `scripts/` e `tests/`

---

### `scripts/` - Scripts Executáveis
Scripts principais que executam funcionalidades do sistema.

#### Scripts Principais:
- **`main.py`**: Sistema de detecção básico (sem ESP32)
- **`main_with_esp32.py`**: Sistema completo com integração ESP32
- **`train_model.py`**: Treinamento do modelo com logging automático

#### Scripts de Preparação:
- **`prepare_ur_fall.py`**: Processa dataset UR Fall (PNG → vídeos)
- **`collect_videos.py`**: Coleta dados via webcam

#### Scripts Auxiliares:
- **`simulate_fall.py`**: Simulação visual de detecção de quedas
- **`generate_example_logs.py`**: Gera logs de exemplo
- **`generate_topic_images.py`**: Gera diagramas para documentação

**Uso:** Execute diretamente: `python scripts/nome_script.py`

---

### `hardware/` - Código Hardware
Código para microcontroladores e dispositivos embarcados.

- **`esp32_fall_alert.ino`**: Código Arduino para ESP32
  - Recebe alertas via Serial/MQTT
  - Dispara buzzer e LEDs quando queda detectada

**Uso:** Abra no Arduino IDE e faça upload para ESP32

---

### `tests/` - Testes
Scripts de teste e validação.

- **`test_esp32.py`**: Testa conexão e comunicação com ESP32

**Uso:** Execute para validar integração: `python tests/test_esp32.py`

---

### `configs/` - Configurações
Configurações centralizadas do projeto.

- **`config.py`**: 
  - Caminhos de diretórios
  - Parâmetros do modelo
  - Configurações ESP32
  - Hiperparâmetros

**Uso:** Importado por scripts: `from configs.config import ...`

---

### `mobile/` - Aplicativo Mobile
Aplicativo React Native (Expo SDK 54, TypeScript) para receber alertas de queda via MQTT e notificar o cuidador.

```
mobile/
├── GUIA_APP_MOBILE.md              # Guia de setup e execução
├── docs/
│   ├── RELATORIO_FASE_MOBILE.md    # Relatório da fase mobile (PAIC)
│   └── images/                     # Diagramas e mockups
└── FallDetectApp/                  # Projeto Expo
    ├── App.tsx                     # Entrada + navegação + provider
    └── src/
        ├── context/AppContext.tsx   # Estado global (useReducer)
        ├── services/               # MqttService, AlarmService, EventStorage
        ├── screens/                # Dashboard, Alarme, Histórico, Configurações
        ├── components/             # StatusBadge, EventCard
        ├── types/                  # Tipos TypeScript (payloads, eventos)
        └── theme/                  # Paleta dark theme
```

**Funcionalidades:** conexão MQTT via WebSocket, alarme sonoro + vibração, histórico persistido no AsyncStorage, filtros por tipo, discagem de emergência, limiar de confiança configurável.

**Uso:** `cd mobile/FallDetectApp && npm install && npx expo start --lan`

---

### `docs/` - Documentação
Documentação geral do projeto.

- **`ESP32_INTEGRATION.md`**: Guia completo de integração ESP32
- **`ESTRUTURA_PROJETO.md`**: Este arquivo
- **`Relatorio-Mes4-PAIC.md`**: Relatório mensal PAIC (implementação da arquitetura neural)
- **`RESULTADOS_OBTIDOS.txt`**: Resultados do projeto

---

### `assets/` - Recursos Visuais
Imagens, diagramas e recursos visuais.

- **`otimizacao_modelo.png`**: Diagrama de otimização
- **`protocolos_alerta_fcm.png`**: Arquitetura de alertas
- **`documentacao_tecnica.png`**: Estrutura de documentação

---

### `data/` - Dados
Dados de treinamento e validação.

```
data/
└── raw/
    ├── Fall/          # Vídeos de quedas (.avi)
    └── Normal/        # Vídeos de atividades normais (.avi)
```

**Nota:** Vídeos não são versionados (muito grandes). Veja `.gitignore`

---

### `models/` - Modelos Treinados
Modelos de deep learning treinados.

- **`fall_model_cnn_lstm.h5`**: Modelo principal (gerado por `train_model.py`)

**Nota:** Modelos não são versionados (muito grandes). Veja `.gitignore`

---

### `logs/` - Logs de Performance
Logs de cada execução de treinamento.

```
logs/
├── README.md                    # Documentação dos logs
└── run-YYYYMMDD-HHMMSS/         # Pasta por execução
    ├── run_metadata.json
    ├── history.json
    ├── history.csv
    ├── curves.png
    ├── confusion_matrix.png
    └── ...
```

**Nota:** Conteúdo dos logs não é versionado, apenas estrutura. Veja `.gitignore`

---

## 🔄 Fluxo de Trabalho Típico

### 1. Preparação de Dados
```bash
python scripts/prepare_ur_fall.py
# ou
python scripts/collect_videos.py
```

### 2. Treinamento
```bash
python scripts/train_model.py
# Logs salvos em logs/run-YYYYMMDD-HHMMSS/
```

### 3. Execução
```bash
# Básico
python scripts/main.py

# Com ESP32
python scripts/main_with_esp32.py
```

### 4. Testes
```bash
python tests/test_esp32.py
```

---

## 📝 Convenções

### Imports
Scripts em `scripts/` devem importar assim:
```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from configs.config import ...
from src.model import ...
```

### Configurações
Sempre use `configs/config.py` para configurações. Não hardcode valores.

### Nomes de Arquivos
- Scripts: `snake_case.py`
- Módulos: `snake_case.py`
- Configs: `snake_case.py`
- Docs: `UPPER_CASE.md` ou `Title_Case.md`

---

## 🚀 Adicionando Novos Arquivos

### Novo Script Executável
1. Crie em `scripts/nome_script.py`
2. Use imports padrão (veja acima)
3. Documente no README.md

### Novo Módulo
1. Crie em `src/nome_modulo.py`
2. Documente com docstrings
3. Adicione testes em `tests/` se necessário

### Nova Configuração
1. Adicione em `configs/config.py`
2. Documente o propósito
3. Atualize este documento se necessário

---

## 📚 Referências

- [README Principal](../README.md)
- [Integração ESP32](ESP32_INTEGRATION.md)
- [Guia do App Mobile](../mobile/GUIA_APP_MOBILE.md)
- [Relatório da Fase Mobile](../mobile/docs/RELATORIO_FASE_MOBILE.md)
- [Logs de Performance](../logs/README.md)

---

**Última atualização:** Março 2026
