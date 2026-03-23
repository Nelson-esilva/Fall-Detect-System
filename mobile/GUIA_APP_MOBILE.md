# 📱 Guia: App Mobile para Detecção de Quedas

## Situação Atual do Projeto

O sistema **Fall-Detect-System** possui:
- ✅ Modelo CNN+LSTM treinado para detectar quedas via vídeo
- ✅ Comunicação com ESP32 via Serial/MQTT
- ✅ Hardware Arduino (buzzer + LEDs) para alertas locais
- ✅ **App mobile funcional** para receber alertas e notificar cuidadores

---

## 📐 Arquitetura

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PC + Câmera │     │   Broker     │     │  App Mobile  │
│  (Python)    │────►│   MQTT       │────►│  (React      │
│  CNN+LSTM    │     │  (Mosquitto) │     │   Native)    │
│  Detecção    │     │              │     │  Alarme!     │
└──────────────┘     └──────┬───────┘     └──────────────┘
                  porta 1883│ porta 9001 (WebSocket)
                     ┌──────┴──────┐
                     │   ESP32     │
                     │  (Buzzer +  │
                     │   LEDs)     │
                     └─────────────┘
```

**Fluxo:**
1. PC detecta queda via câmera → publica mensagem MQTT (porta 1883)
2. Broker Mosquitto distribui para todos os assinantes
3. App mobile recebe via WebSocket (porta 9001) → alarme sonoro + vibração
4. ESP32 recebe via MQTT (porta 1883) → buzzer + LEDs

---

## 🛠️ Stack Tecnológico

| Componente | Tecnologia | Versão |
|---|---|---|
| **Framework** | React Native | 0.81.5 |
| **Toolchain** | Expo | SDK 54 |
| **Linguagem** | TypeScript | 5.x |
| **Comunicação** | MQTT via WebSocket | mqtt.js 5.x |
| **Broker** | Eclipse Mosquitto | 2.x |

---

## 📁 Estrutura do App

```
mobile/FallDetectApp/
├── App.tsx                      # Entrada + navegação + AppProvider
├── assets/
│   └── alarm.wav                # Som do alarme (3s, 880/660 Hz)
└── src/
    ├── context/
    │   └── AppContext.tsx        # Estado global (MQTT, alarme, eventos, settings)
    ├── services/
    │   ├── MqttService.ts       # Conexão WebSocket com broker Mosquitto
    │   ├── AlarmService.ts      # Som em loop + vibração contínua
    │   └── EventStorage.ts      # Persistência no AsyncStorage
    ├── screens/
    │   ├── DashboardScreen.tsx   # Status MQTT, estatísticas, botão teste
    │   ├── AlarmScreen.tsx       # Alarme ativo/inativo, confirmar/descartar
    │   ├── HistoryScreen.tsx     # Lista de eventos com filtros
    │   └── SettingsScreen.tsx    # Configuração do broker e alarme
    ├── components/
    │   ├── StatusBadge.tsx       # Indicador de conexão
    │   └── EventCard.tsx         # Card de evento
    ├── types/
    │   └── index.ts             # Tipos TypeScript
    └── theme/
        └── colors.ts            # Paleta dark theme
```

---

## 📋 Funcionalidades Implementadas

### Tela 1 — Dashboard
- Status da conexão MQTT com indicador visual (verde/vermelho/amarelo)
- Botão Conectar/Desconectar
- Card do último evento com timestamp e confiança
- Contadores de alertas (hoje / esta semana)
- Botão "Testar Alarme" (dispara alarme local sem broker)

### Tela 2 — Alarme de Queda
- Ativação automática ao receber `FALL_DETECTED` via MQTT
- Alarme sonoro em loop + vibração contínua
- Confiança da detecção e timestamp
- Botão "Confirmar Queda" → registra e oferece ligar emergência
- Botão "Falso Alarme" → silencia e registra como falso positivo
- Botão "Ligar Emergência" → discagem direta
- Badge "!" na aba enquanto alarme ativo

### Tela 3 — Histórico
- Lista cronológica persistida no AsyncStorage (até 200 eventos)
- Filtros: Todos, Confirmados, Falso Alarme, Pendentes, Testes
- Botão "Limpar" para apagar histórico

### Tela 4 — Configurações
- Endereço do broker MQTT, porta WebSocket, tópico
- Número de emergência
- Limiar de confiança para acionar alarme (padrão: 70%)
- Toggle de notificações
- Configurações persistidas no AsyncStorage

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
cd mobile/FallDetectApp
npm install
```

### 2. Configurar o broker MQTT (Mosquitto)

```bash
# Instalar
sudo apt install mosquitto mosquitto-clients

# Configurar WebSocket
sudo tee /etc/mosquitto/conf.d/websocket.conf > /dev/null << 'EOF'
listener 1883
allow_anonymous true

listener 9001
protocol websockets
allow_anonymous true
EOF

# Reiniciar
sudo systemctl restart mosquitto
```

### 3. Descobrir o IP do PC

```bash
hostname -I | awk '{print $1}'
```

### 4. Rodar o app no celular

```bash
npx expo start --lan
# Escanear o QR code com o app Expo Go (v54) no celular
```

### 5. No app: Configurações → colocar o IP do PC → Dashboard → Conectar

### 6. Testar com simulação de queda

```bash
# Simular uma queda via terminal:
mosquitto_pub -t "fall_detection/alerts" \
  -m '{"alert":"FALL_DETECTED","confidence":0.92,"timestamp":"2026-03-22T15:30:00","metadata":{"frame_id":42,"model":"CNN-LSTM"}}'

# Ou usar o script existente do projeto (requer MQTT ativado no config.py):
python3 scripts/simulate_fall.py
```

---

## 🔧 Integração com o Backend Python

O `src/esp32_interface.py` já suporta MQTT. Para ativar:

1. Editar `configs/config.py`:
   ```python
   ESP32_CONNECTION_TYPE = "mqtt"
   ESP32_BROKER = "192.168.x.x"  # IP do PC na rede local
   ESP32_PORT = 1883
   ESP32_TOPIC = "fall_detection/alerts"
   ```

2. Adicionar ao import em `scripts/main_with_esp32.py`:
   ```python
   from configs.config import (
       MODEL_PATH, IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH, CLASSES,
       ESP32_CONNECTION_TYPE, ESP32_PORT, ESP32_BAUDRATE,
       ESP32_BROKER, ESP32_TOPIC  # adicionar estas duas
   )
   ```

3. O app e o ESP32 assinam o mesmo tópico — ambos recebem o alerta simultaneamente.

---

## ⏭️ Próximos Passos (Etapa 4)

1. Notificações push locais (`expo-notifications`) para alertar quando o app está em segundo plano
2. Animações pulsantes na tela de alarme
3. Testes integrados com o sistema completo (PC + ESP32 + App via MQTT)
4. Build de produção (APK/AAB) para instalação sem Expo Go
