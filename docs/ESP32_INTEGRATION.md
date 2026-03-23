# Integração ESP32 - Sistema de Alerta de Quedas

Este documento descreve como integrar o sistema de detecção de quedas com um ESP32 para disparar alertas locais (buzzer, LEDs) quando uma queda é detectada.

## 📋 Visão Geral

O sistema de detecção de quedas (PC/Python) detecta quedas através de visão computacional e envia alertas para o ESP32, que por sua vez dispara:
- **Buzzer** (alerta sonoro)
- **LED vermelho** (alerta visual)
- **LED verde** (status do sistema)

## 🔌 Hardware Necessário

### ESP32
- Qualquer variante (ESP32 DevKit, ESP32-WROOM, etc.)

### Componentes
- **Buzzer passivo** (5V) → GPIO 18
- **LED vermelho** (com resistor 220Ω) → GPIO 19
- **LED verde** (com resistor 220Ω) → GPIO 21
- **Botão de teste** (opcional) → GPIO 0

### Esquema de Conexão

```
ESP32          Componente
-----          ----------
GPIO 18  ────> Buzzer (+)
GND      ────> Buzzer (-)
GPIO 19  ────> LED Vermelho (anodo) ──[220Ω]──> GND
GPIO 21  ────> LED Verde (anodo) ──[220Ω]──> GND
GPIO 0   ────> Botão ──> GND (pull-up interno)
```

## 📡 Métodos de Comunicação

### 1. Serial (USB) - Recomendado para Testes

**Vantagens:**
- Simples de configurar
- Não precisa de WiFi
- Ideal para desenvolvimento

**Desvantagens:**
- Requer cabo USB conectado
- Distância limitada

### 2. WiFi/MQTT - Recomendado para Produção

**Vantagens:**
- Sem fios (WiFi)
- Pode estar em outro cômodo
- Escalável (múltiplos ESP32s)

**Desvantagens:**
- Requer configuração de rede
- Mais complexo

## 🚀 Instalação e Configuração

### Passo 1: Instalar Dependências Python

```bash
pip install pyserial paho-mqtt
```

### Passo 2: Carregar Código no ESP32

1. Abra o Arduino IDE
2. Instale a biblioteca **ArduinoJson** (Tools → Manage Libraries → "ArduinoJson")
3. Abra o arquivo `esp32_fall_alert.ino`
4. Selecione a placa: **Tools → Board → ESP32 Dev Module**
5. Selecione a porta: **Tools → Port → COM3** (ou sua porta)
6. Faça upload do código

### Passo 3: Configurar Python

Edite `configs/config.py` e ajuste as configurações:

```python
# Para Serial (USB)
ESP32_CONNECTION_TYPE = "serial"
ESP32_PORT = "COM3"  # Windows: COM3, Linux: /dev/ttyUSB0, Mac: /dev/cu.usbserial-*
ESP32_BAUDRATE = 115200

# Para MQTT (WiFi) - descomente e configure:
# ESP32_CONNECTION_TYPE = "mqtt"
# ESP32_BROKER = "192.168.1.100"  # IP do broker MQTT
# ESP32_TOPIC = "fall_detection/alerts"
```

### Passo 4: Executar

```bash
python scripts/main_with_esp32.py
```

## 🔧 Configuração MQTT (Opcional)

Se quiser usar WiFi/MQTT:

### 1. Instalar Broker MQTT no PC

**Opção A: Mosquitto (Windows)**
```bash
# Baixe de: https://mosquitto.org/download/
# Instale e inicie o serviço
```

**Opção B: Docker**
```bash
docker run -it -p 1883:1883 eclipse-mosquitto
```

### 2. Configurar ESP32 para MQTT

No arquivo `esp32_fall_alert.ino`, descomente as seções MQTT e configure:

```cpp
#define WIFI_SSID "SEU_WIFI"
#define WIFI_PASSWORD "SUA_SENHA"
#define MQTT_BROKER "192.168.1.100"  // IP do PC com broker
#define MQTT_PORT 1883
#define MQTT_TOPIC "fall_detection/alerts"
```

### 3. Instalar Biblioteca MQTT no ESP32

No Arduino IDE:
- **Tools → Manage Libraries → "PubSubClient"**

## 🧪 Testes

### Teste 1: Alerta Manual (Botão)

Pressione o botão no ESP32 (GPIO 0). O buzzer deve tocar e o LED vermelho piscar por 10 segundos.

### Teste 2: Alerta via Python

Execute o sistema e pressione **'t'** no teclado. Um alerta de teste será enviado ao ESP32.

### Teste 3: Detecção Real

Execute `scripts/main_with_esp32.py` e simule uma queda. O sistema deve:
1. Detectar a queda
2. Enviar alerta ao ESP32
3. ESP32 disparar buzzer e LED

## 📊 Formato das Mensagens

### Serial/MQTT (JSON)

```json
{
  "alert": "FALL_DETECTED",
  "confidence": 0.95,
  "timestamp": "2025-01-29T10:30:45.123456",
  "metadata": {
    "frame_id": 20,
    "model": "CNN-LSTM"
  }
}
```

### Alerta de Teste

```json
{
  "alert": "TEST",
  "confidence": 0.95,
  "timestamp": "2025-01-29T10:30:45.123456",
  "metadata": {
    "type": "test"
  }
}
```

## 🐛 Troubleshooting

### ESP32 não conecta via Serial

1. Verifique a porta: `COM3`, `/dev/ttyUSB0`, etc.
2. Verifique se outra aplicação está usando a porta
3. Tente outra porta USB
4. Verifique se o driver USB-Serial está instalado

### ESP32 não recebe alertas

1. Verifique se o LED verde está ligado (sistema OK)
2. Abra o Serial Monitor do Arduino IDE (115200 baud) para ver mensagens
3. Teste com `esp32.send_test_alert()` no Python

### MQTT não funciona

1. Verifique se o broker está rodando
2. Verifique se ESP32 está conectado ao WiFi
3. Verifique firewall (porta 1883)
4. Teste com cliente MQTT (ex: MQTT.fx)

## 📝 Código de Exemplo

### Python - Enviar Alerta Manual

```python
from src.esp32_interface import create_esp32_interface

# Conectar
esp32 = create_esp32_interface("serial", port="COM3")

# Enviar alerta
esp32.send_alert(confidence=0.95)

# Desconectar
esp32.disconnect()
```

### Python - Usar com Callback

```python
def on_alert_sent(alert_data):
    print(f"Alerta enviado: {alert_data}")

esp32 = create_esp32_interface("serial", port="COM3")
esp32.set_alert_callback(on_alert_sent)
esp32.send_alert(confidence=0.90)
```

## 🔒 Segurança

- **Produção**: Use MQTT com autenticação
- **WiFi**: Use WPA2 ou superior
- **MQTT**: Configure usuário/senha no broker

## 📱 Integração com App Mobile

Com o modo MQTT ativo, o ESP32 e o aplicativo mobile recebem alertas simultaneamente pelo mesmo broker Mosquitto. O app se conecta via WebSocket (porta 9001) enquanto o ESP32 usa MQTT nativo (porta 1883).

Para configurar o app mobile e o broker com WebSocket, veja o [Guia do App Mobile](../mobile/GUIA_APP_MOBILE.md).

## 📚 Referências

- [ESP32 Arduino Core](https://github.com/espressif/arduino-esp32)
- [ArduinoJson](https://arduinojson.org/)
- [PubSubClient (MQTT)](https://github.com/knolleary/pubsubclient)
- [pyserial](https://pyserial.readthedocs.io/)
- [paho-mqtt](https://pypi.org/project/paho-mqtt/)

## ✅ Checklist de Deploy

- [ ] ESP32 programado com `esp32_fall_alert.ino`
- [ ] Hardware conectado corretamente
- [ ] Dependências Python instaladas
- [ ] Porta Serial configurada corretamente
- [ ] Teste de alerta manual funcionando
- [ ] Teste de alerta via Python funcionando
- [ ] Sistema de detecção integrado
- [ ] Documentação atualizada

---

**Status**: ✅ Código validado e pronto para deploy
