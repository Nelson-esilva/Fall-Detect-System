# Sistema de Detecção de Quedas com IA

Sistema de detecção de quedas em tempo real utilizando **Deep Learning (CNN + LSTM)**, visão computacional, alertas via **ESP32** e **aplicativo mobile** para notificação de cuidadores.

**Projeto de Iniciação Científica — PAIC/FAPEAM — Universidade do Estado do Amazonas (UEA)**

---

## Sobre o Projeto

Este sistema detecta quedas através de câmeras de vídeo e distribui alertas simultaneamente para um microcontrolador ESP32 (alarme local) e um aplicativo mobile (alarme remoto), utilizando o protocolo MQTT como camada de comunicação.

### Arquitetura

```
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  PC + Câmera     │        │  Broker MQTT     │        │  App Mobile      │
│                  │  pub   │  (Mosquitto)     │  sub   │  (React Native)  │
│  MobileNetV2     │───────►│                  │───────►│  Alarme sonoro   │
│  + LSTM          │        │  porta 1883      │        │  + vibração      │
│  Detecção        │        │  porta 9001 (WS) │        │  + emergência    │
└──────────────────┘        └────────┬─────────┘        └──────────────────┘
                                     │ sub
                            ┌────────▼─────────┐
                            │  ESP32           │
                            │  Buzzer + LEDs   │
                            └──────────────────┘
```

### Modelo de IA

O modelo utiliza uma arquitetura híbrida CNN-LSTM para analisar sequências temporais de vídeo:

1. **MobileNetV2 (CNN)** — extrai características visuais de cada frame (transfer learning, pesos congelados do ImageNet)
2. **TimeDistributed** — aplica a CNN nos 20 frames da janela temporal individualmente
3. **LSTM (64 unidades)** — analisa a sequência de vetores de características para identificar o padrão de movimento de queda
4. **Dense (sigmoid)** — classificação binária: Normal ou Queda

O modelo foi treinado no **UR Fall Detection Dataset** (30 sequências de quedas + 40 de atividades diárias).

---

## Estrutura do Projeto

```
Fall-Detect-System/
├── src/                              # Módulos Python reutilizáveis
│   ├── model.py                      #   Arquitetura CNN-LSTM
│   └── esp32_interface.py            #   Interface Serial/MQTT com ESP32
│
├── scripts/                          # Scripts executáveis
│   ├── main.py                       #   Detecção em tempo real (básico)
│   ├── main_with_esp32.py            #   Detecção + alertas ESP32/MQTT
│   ├── train_model.py                #   Treinamento do modelo
│   ├── prepare_ur_fall.py            #   Processamento do dataset UR Fall
│   ├── collect_videos.py             #   Coleta de dados via webcam
│   ├── simulate_fall.py              #   Simulação visual de detecção
│   ├── generate_example_logs.py      #   Geração de logs de exemplo
│   └── generate_topic_images.py      #   Geração de diagramas
│
├── hardware/                         # Firmware do microcontrolador
│   └── esp32_fall_alert.ino          #   Arduino — buzzer + LEDs via Serial/MQTT
│
├── mobile/                           # Aplicativo mobile
│   ├── GUIA_APP_MOBILE.md            #   Guia de setup e execução
│   ├── docs/                         #   Relatório e imagens da fase mobile
│   └── FallDetectApp/                #   Projeto React Native (Expo SDK 54)
│       ├── App.tsx                   #     Entrada + navegação + provider
│       └── src/
│           ├── context/              #     Estado global (MQTT, alarme, eventos)
│           ├── services/             #     MqttService, AlarmService, EventStorage
│           ├── screens/              #     Dashboard, Alarme, Histórico, Configurações
│           ├── components/           #     StatusBadge, EventCard
│           ├── types/                #     Tipos TypeScript
│           └── theme/                #     Paleta dark theme
│
├── configs/
│   └── config.py                     # Configurações centralizadas
│
├── tests/
│   └── test_esp32.py                 # Teste de conexão ESP32
│
├── docs/                             # Documentação geral
│   ├── ESP32_INTEGRATION.md          #   Guia completo de integração ESP32
│   ├── ESTRUTURA_PROJETO.md          #   Detalhamento da estrutura
│   ├── Relatorio-Mes4-PAIC.md        #   Relatório mensal PAIC
│   └── RESULTADOS_OBTIDOS.txt        #   Resultados do projeto
│
├── assets/                           # Diagramas e imagens
├── data/raw/                         # Vídeos de treino (Fall/ e Normal/)
├── models/                           # Modelos treinados (.h5)
├── logs/                             # Logs de treinamento por execução
│
├── requirements.txt                  # Dependências Python
└── README.md
```

---

## Instalação e Execução

### Pré-requisitos

- Python 3.10+
- Node.js 18+ (para o app mobile)
- Arduino IDE (para o ESP32)

### 1. Detecção de Quedas (Python)

```bash
# Instalar dependências
pip install -r requirements.txt

# Preparar dataset UR Fall Detection
# Baixe de: https://fenix.ur.edu.pl/mkepski/ds/uf.html
# Extraia em UR_Fall_Downloads/ e execute:
python scripts/prepare_ur_fall.py

# Treinar o modelo
python scripts/train_model.py

# Executar detecção em tempo real
python scripts/main.py
```

### 2. Integração ESP32

O ESP32 recebe alertas do PC via Serial (USB) ou MQTT (WiFi) e dispara alarmes locais com buzzer e LEDs.

```bash
# 1. Carregue hardware/esp32_fall_alert.ino no ESP32 via Arduino IDE
#    (instale a biblioteca ArduinoJson)

# 2. Configure a porta em configs/config.py
#    ESP32_PORT = "/dev/ttyUSB0"  # ou COM3 no Windows

# 3. Teste a conexão
python tests/test_esp32.py

# 4. Execute detecção com alertas ESP32
python scripts/main_with_esp32.py
```

Documentação completa: [docs/ESP32_INTEGRATION.md](docs/ESP32_INTEGRATION.md)

### 3. Aplicativo Mobile

O app React Native recebe alertas via MQTT (WebSocket) e notifica o cuidador com alarme sonoro, vibração e opção de ligar para emergência.

```bash
# 1. Instalar e configurar o broker MQTT
sudo apt install mosquitto mosquitto-clients
sudo tee /etc/mosquitto/conf.d/websocket.conf > /dev/null << 'EOF'
listener 1883
allow_anonymous true
listener 9001
protocol websockets
allow_anonymous true
EOF
sudo systemctl restart mosquitto

# 2. Instalar dependências do app
cd mobile/FallDetectApp
npm install

# 3. Rodar no celular (Expo Go)
npx expo start --lan
# Escaneie o QR code com o Expo Go no celular

# 4. No app: Configurações → IP do PC → Dashboard → Conectar
```

Guia completo: [mobile/GUIA_APP_MOBILE.md](mobile/GUIA_APP_MOBILE.md)

### 4. Teste integrado (PC + ESP32 + App)

Com o Mosquitto rodando e o app conectado, simule uma queda:

```bash
mosquitto_pub -t "fall_detection/alerts" \
  -m '{"alert":"FALL_DETECTED","confidence":0.92,"timestamp":"2026-03-22T15:30:00","metadata":{"frame_id":42,"model":"CNN-LSTM"}}'
```

O ESP32 dispara buzzer + LEDs e o app exibe a tela de alarme simultaneamente.

---

## Stack Tecnológico

| Camada | Tecnologia | Versão |
|---|---|---|
| Modelo de IA | TensorFlow/Keras (MobileNetV2 + LSTM) | 2.x |
| Visão Computacional | OpenCV | 4.x |
| Backend | Python | 3.10+ |
| Comunicação | MQTT (Eclipse Mosquitto) | 2.x |
| App Mobile | React Native + Expo (TypeScript) | RN 0.81 / SDK 54 |
| Hardware | ESP32 + Arduino (buzzer, LEDs) | — |

---

## Protocolo de Comunicação

O sistema usa MQTT no modelo publish/subscribe. O PC publica alertas no tópico `fall_detection/alerts` e todos os assinantes (ESP32 + app) recebem simultaneamente.

**Payload JSON:**
```json
{
  "alert": "FALL_DETECTED",
  "confidence": 0.95,
  "timestamp": "2026-03-22T15:30:45.123456",
  "metadata": {
    "frame_id": 1234,
    "model": "CNN-LSTM"
  }
}
```

| Assinante | Transporte | Porta |
|-----------|-----------|-------|
| ESP32 | MQTT nativo | 1883 |
| App Mobile | MQTT via WebSocket | 9001 |

---

## Configuração

As configurações centralizadas estão em `configs/config.py`:

```python
# Modelo
MODEL_PATH = 'models/fall_model_cnn_lstm.h5'
IMG_HEIGHT, IMG_WIDTH = 224, 224
SEQUENCE_LENGTH = 20

# ESP32
ESP32_CONNECTION_TYPE = "serial"  # "serial" ou "mqtt"
ESP32_PORT = "COM3"
ESP32_BAUDRATE = 115200

# MQTT (para usar com app mobile + ESP32 via WiFi)
# ESP32_BROKER = "192.168.x.x"
# ESP32_TOPIC = "fall_detection/alerts"
```

---

## Dataset

**UR Fall Detection Dataset**
- 30 sequências de quedas + 40 de atividades diárias (ADL)
- Câmeras RGB + Profundidade + Acelerômetro
- Processamento: sliding window com passo de 10 frames

> Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. *Computer Methods and Programs in Biomedicine*, 117(3), 489-501.

---

## Logs de Treinamento

Cada execução de `train_model.py` gera uma pasta `logs/run-YYYYMMDD-HHMMSS/` com:

| Arquivo | Conteúdo |
|---------|----------|
| `run_metadata.json` | Configurações e hiperparâmetros |
| `history.json` / `history.csv` | Métricas por época |
| `curves.png` | Gráficos de loss e accuracy |
| `confusion_matrix.png` | Matriz de confusão |
| `classification_report.txt` | Precisão, recall e F1-score |
| `final_metrics.json` | Métricas finais no conjunto de teste |

Detalhes em [logs/README.md](logs/README.md).

---

## Documentação

| Documento | Conteúdo |
|-----------|----------|
| [docs/ESP32_INTEGRATION.md](docs/ESP32_INTEGRATION.md) | Guia completo de integração ESP32 (hardware, serial, MQTT, troubleshooting) |
| [docs/ESTRUTURA_PROJETO.md](docs/ESTRUTURA_PROJETO.md) | Detalhamento da estrutura de pastas e convenções |
| [docs/Relatorio-Mes4-PAIC.md](docs/Relatorio-Mes4-PAIC.md) | Relatório mensal PAIC — implementação da arquitetura neural |
| [mobile/GUIA_APP_MOBILE.md](mobile/GUIA_APP_MOBILE.md) | Guia de setup e execução do app mobile |
| [mobile/docs/RELATORIO_FASE_MOBILE.md](mobile/docs/RELATORIO_FASE_MOBILE.md) | Relatório da fase mobile — arquitetura, implementação, testes |
| [logs/README.md](logs/README.md) | Estrutura e interpretação dos logs de treinamento |

---

## Autor

**Nelson Emeliano Silva**  
Orientador: Prof. Angilberto Muniz Ferreira Sobrinho  
Universidade do Estado do Amazonas — UEA

---

## Licença

Este projeto é para fins acadêmicos e de pesquisa (PAIC/FAPEAM).
