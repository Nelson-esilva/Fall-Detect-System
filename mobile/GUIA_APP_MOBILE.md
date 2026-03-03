# 📱 Guia: App Mobile para Detecção de Quedas

## Situação Atual do Projeto

Seu sistema **Fall-Detect-System** já possui:
- ✅ Modelo CNN+LSTM treinado para detectar quedas via vídeo
- ✅ Comunicação com ESP32 via Serial/MQTT
- ✅ Hardware Arduino (buzzer + LEDs) para alertas locais

**O que falta:** um app mobile para receber os alertas e notificar cuidadores/familiares.

---

## 🏗️ Abordagem Recomendada: React Native + Expo

| Aspecto | Detalhe |
|---|---|
| **Framework** | React Native com Expo |
| **Linguagem** | JavaScript/TypeScript |
| **Plataformas** | Android + iOS com um único código |
| **Comunicação** | MQTT (já suportado no seu backend) |
| **Notificações** | Firebase Cloud Messaging (FCM) |
| **Complexidade** | Baixa-Média |

### Por que React Native + Expo?
- **Rápido para prototipar** — ideal para projeto acadêmico/PAIC
- **Um código → dois apps** (Android + iOS)
- **Expo simplifica** build, deploy e testes no celular
- **Enorme comunidade** e bibliotecas prontas

---

## 📐 Arquitetura Proposta

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PC + Câmera │     │   Broker     │     │  App Mobile  │
│  (Python)    │────►│   MQTT       │────►│  (React      │
│  CNN+LSTM    │     │  (Mosquitto) │     │   Native)    │
│  Detecção    │     │              │     │  Alarme!     │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     │   ESP32     │
                     │  (Buzzer +  │
                     │   LEDs)     │
                     └─────────────┘
```

**Fluxo:**
1. PC detecta queda via câmera → publica mensagem MQTT
2. Broker MQTT distribui para todos os assinantes
3. App mobile recebe e dispara alarme sonoro + notificação push
4. ESP32 também recebe e dispara alarme local (já funciona)

---

## 🛠️ O Que Você Precisa

### Software
| Item | Para que serve |
|---|---|
| **Node.js 18+** | Runtime do React Native (✅ já instalado: v20.20.0) |
| **Expo CLI** | `npm install -g expo-cli` |
| **Expo Go** (app no celular) | Testar no celular sem build |
| **Broker MQTT** | Mosquitto (no PC ou servidor) |

### Dependências do App (npm)
| Pacote | Função |
|---|---|
| `mqtt` | Conectar ao broker MQTT |
| `expo-notifications` | Notificações push |
| `expo-av` | Tocar alarme sonoro |
| `@react-navigation/native` | Navegação entre telas |

### Hardware/Infra
- Celular Android ou iOS para testes
- PC e celular na **mesma rede WiFi** (para MQTT local)
- Opcionalmente: broker MQTT na nuvem (HiveMQ, CloudMQTT)

---

## 📋 Funcionalidades do App (MVP)

### Tela 1 — Dashboard
- Status da conexão MQTT (🟢 conectado / 🔴 desconectado)
- Último evento detectado (timestamp + confiança)
- Botão de teste de alarme

### Tela 2 — Alarme de Queda
- **Alarme sonoro** alto (vibração + som)
- Nível de confiança da detecção
- Botões: "Confirmar Queda" / "Falso Alarme"
- Botão "Ligar para Emergência" (discagem direta)

### Tela 3 — Histórico
- Lista de eventos com data/hora e confiança
- Filtro por período

### Tela 4 — Configurações
- Endereço do broker MQTT
- Número de emergência
- Volume do alarme
- Limiar de confiança para alarme

---

## 🔧 Mudanças no Backend Python (Mínimas)

O código em `src/esp32_interface.py` já suporta MQTT! Basta:

1. **Instalar e configurar o Mosquitto** (broker MQTT) no PC:
   ```bash
   sudo apt install mosquitto mosquitto-clients
   sudo systemctl enable mosquitto
   sudo systemctl start mosquitto
   ```

2. **Atualizar** `configs/config.py`:
   ```python
   ESP32_CONNECTION_TYPE = "mqtt"
   ESP32_BROKER = "192.168.x.x"  # IP do seu PC na rede local
   ESP32_TOPIC = "fall_detection/alerts"
   ```

3. O app e o ESP32 assinam o mesmo tópico — ambos recebem o alerta simultaneamente.

---

## 🚀 Passo a Passo para Começar

### 1. Instalar Expo CLI
```bash
npm install -g expo-cli
```

### 2. Criar o App (dentro desta pasta `mobile/`)
```bash
cd mobile/
npx create-expo-app FallDetectApp
cd FallDetectApp
```

### 3. Instalar dependências
```bash
npx expo install expo-av expo-notifications
npm install mqtt @react-navigation/native @react-navigation/native-stack
```

### 4. Configurar broker MQTT no PC
```bash
sudo apt install mosquitto mosquitto-clients
# Testar publicação:
mosquitto_pub -t "fall_detection/alerts" -m '{"alert":"FALL_DETECTED","confidence":0.95}'
```

### 5. Testar no celular
```bash
npx expo start
# Escanear o QR code com o app Expo Go no celular
```

### 6. Testar com simulação de queda
```bash
# No terminal do PC (já existe no projeto!):
python scripts/simulate_fall.py
```

---

## ⚡ Alternativas ao React Native

| Framework | Prós | Contras |
|---|---|---|
| **Flutter** (Dart) | Performance excelente, UI bonita | Precisa aprender Dart |
| **Kotlin/Swift nativo** | Máxima performance | Código separado para cada OS |
| **PWA (Web App)** | Mais simples, sem instalação | Limitações em som/notificação de fundo |

> **💡 Dica:** Para um projeto acadêmico/PAIC, **React Native + Expo** é a melhor relação custo-benefício.

---

## 📁 Estrutura Final do Projeto

```
Fall-Detect-System/
├── src/                    # Python - Modelo e interface ESP32
├── scripts/                # Python - Scripts de detecção
├── hardware/               # Arduino - Código do ESP32
├── mobile/                 # ← NOVO
│   ├── GUIA_APP_MOBILE.md  # Este guia
│   └── FallDetectApp/      # App React Native (Expo)
│       ├── App.js
│       ├── package.json
│       └── ...
├── configs/                # Configurações Python
├── models/                 # Modelos treinados
└── README.md
```
