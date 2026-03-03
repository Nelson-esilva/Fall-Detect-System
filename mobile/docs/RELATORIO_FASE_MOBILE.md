# 📱 Relatório de Atividades — Fase Mobile
## Sistema de Detecção de Quedas com IA

**Projeto:** PAIC/FAPEAM — Universidade do Estado do Amazonas (UEA)  
**Autor:** Nelson Emeliano Silva  
**Orientador:** Prof. Angilberto Muniz Ferreira Sobrinho  
**Data:** Março/2026  
**Fase:** Desenvolvimento do Aplicativo Mobile

---

## 1. Introdução

Este relatório documenta a nova fase do projeto de Detecção de Quedas com IA: o desenvolvimento de um **aplicativo mobile para Android** capaz de receber alertas em tempo real quando o sistema detecta uma queda, emitindo alarmes sonoros e visuais no smartphone do cuidador ou familiar.

### 1.1 Contexto

As fases anteriores do projeto resultaram em:
- Um **modelo híbrido CNN+LSTM** (MobileNetV2 + LSTM) treinado no dataset UR Fall Detection
- Um sistema de **detecção em tempo real** via câmera de vídeo
- **Integração com ESP32** para alarmes locais (buzzer + LEDs) via Serial/MQTT

### 1.2 Objetivo da Fase Mobile

Desenvolver um aplicativo Android que:
1. Receba alertas de queda em tempo real via protocolo MQTT
2. Emita alarmes sonoros e visuais no dispositivo móvel
3. Permita ao cuidador confirmar ou descartar o alerta
4. Mantenha um histórico de eventos para acompanhamento
5. Ofereça acesso rápido à discagem de emergência

---

## 2. Arquitetura do Sistema

### 2.1 Visão Geral

A arquitetura integra o sistema existente (PC + ESP32) com o novo aplicativo mobile através de um broker MQTT centralizado.

![Arquitetura do Sistema](images/arquitetura_sistema.png)

### 2.2 Componentes

| Componente | Tecnologia | Função |
|---|---|---|
| **Módulo de Detecção** | Python + TensorFlow | Processa vídeo da câmera e detecta quedas usando CNN+LSTM |
| **Broker MQTT** | Eclipse Mosquitto | Hub central de mensagens — distribui alertas para todos os assinantes |
| **App Mobile** | React Native + Expo | Recebe alertas e notifica o cuidador com alarme sonoro/visual |
| **Hardware ESP32** | Arduino + WiFi | Dispara alarme local (buzzer + LEDs) |

### 2.3 Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────────┐
│                     FLUXO DE DETECÇÃO                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CAPTURA          2. PROCESSAMENTO       3. CLASSIFICAÇÃO    │
│  ┌──────────┐       ┌──────────────┐       ┌──────────────┐    │
│  │ Câmera   │──────►│ MobileNetV2  │──────►│    LSTM      │    │
│  │ (Webcam) │ Frame │ (Extração de │ Vetor │ (Análise     │    │
│  └──────────┘       │ Features)    │       │  Temporal)   │    │
│                     └──────────────┘       └──────┬───────┘    │
│                                                    │            │
│                                          Queda? (>70%)          │
│                                                    │            │
│  4. DISTRIBUIÇÃO                                   ▼            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              BROKER MQTT (Mosquitto)                  │       │
│  │      Tópico: fall_detection/alerts                    │       │
│  └──────┬───────────────────────────────────┬───────────┘       │
│         │                                   │                    │
│         ▼                                   ▼                    │
│  ┌──────────────┐                   ┌──────────────┐            │
│  │  App Mobile  │                   │    ESP32     │            │
│  │  📱 Alarme   │                   │  🔔 Buzzer   │            │
│  │  📞 Emergência│                  │  🔴 LEDs     │            │
│  └──────────────┘                   └──────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Protocolo de Comunicação — MQTT

O protocolo MQTT (Message Queuing Telemetry Transport) foi escolhido por:
- **Leveza**: ideal para IoT e dispositivos com recursos limitados
- **Baixa latência**: essencial para alertas em tempo real
- **Padrão publish/subscribe**: permite múltiplos assinantes sem acoplamento
- **Já implementado**: o backend Python e o ESP32 já suportam MQTT

**Formato da mensagem de alerta:**
```json
{
  "alert": "FALL_DETECTED",
  "confidence": 0.95,
  "timestamp": "2026-03-03T15:30:45.123456",
  "metadata": {
    "frame_id": 1234,
    "camera_id": "cam01"
  }
}
```

---

## 3. Stack Tecnológico

![Stack Tecnológico](images/stack_tecnologico.png)

### 3.1 Detalhamento

| Camada | Tecnologia | Versão | Justificativa |
|---|---|---|---|
| **Frontend Mobile** | React Native | 0.76+ | Framework cross-platform mais popular, grande comunidade |
| **Toolchain Mobile** | Expo | SDK 52+ | Simplifica build, deploy e testes — ideal para protótipos rápidos |
| **Linguagem Mobile** | JavaScript/TypeScript | ES2022+ | Linguagem universal, baixa curva de aprendizado |
| **Protocolo IoT** | MQTT | v3.1.1 | Leve, baixa latência, padrão pub/sub |
| **Broker MQTT** | Eclipse Mosquitto | 2.x | Open source, leve, amplamente utilizado |
| **Backend IA** | Python 3.10+ | — | Ecossistema maduro para ML/DL |
| **Deep Learning** | TensorFlow/Keras | 2.x | Framework robusto para CNN+LSTM |
| **Visão Computacional** | OpenCV | 4.x | Processamento de vídeo em tempo real |
| **Hardware** | ESP32 + Arduino | — | Microcontrolador WiFi/BT de baixo custo |

### 3.2 Dependências do App Mobile

| Pacote npm | Função |
|---|---|
| `mqtt` | Cliente MQTT para JavaScript — conexão com o broker |
| `expo-av` | Reprodução de áudio (alarme sonoro) |
| `expo-notifications` | Notificações push locais e remotas |
| `expo-haptics` | Vibração do dispositivo |
| `@react-navigation/native` | Navegação entre telas do app |
| `@react-native-async-storage/async-storage` | Armazenamento local (histórico, configurações) |

---

## 4. Design do Aplicativo

### 4.1 Telas Planejadas

![Telas do App Mobile](images/telas_app_mobile.png)

### 4.2 Descrição das Telas

#### Tela 1 — Dashboard (Tela Principal)
- **Indicador de status** da conexão MQTT (conectado/desconectado)
- **Card do último evento** com timestamp e nível de confiança
- **Contadores** de alertas do dia/semana
- **Botão de teste** para verificar funcionamento do alarme

#### Tela 2 — Alarme de Queda
- Ativada automaticamente quando uma queda é detectada
- **Alarme sonoro** em volume máximo + **vibração contínua**
- **Indicador de confiança** da detecção (ex: 95%)
- **Botão "Confirmar Queda"** → registra evento e oferece ligar para emergência
- **Botão "Falso Alarme"** → silencia e registra como falso positivo

#### Tela 3 — Histórico de Eventos
- Lista cronológica de todos os alertas recebidos
- Classificação: Queda Confirmada / Falso Alarme / Não Respondido
- Informações: data, hora, nível de confiança, resposta do cuidador

#### Tela 4 — Configurações
- Endereço IP e porta do broker MQTT
- Tópico MQTT para assinatura
- Número de emergência para discagem rápida
- Limiar de confiança para acionar alarme (padrão: 70%)
- Volume e duração do alarme

---

## 5. Cronograma de Desenvolvimento

![Cronograma de Desenvolvimento](images/cronograma_desenvolvimento.png)

### 5.1 Detalhamento por Semana

| Semana | Fase | Atividades | Entregável |
|---|---|---|---|
| **1** | Configuração & Setup | Instalação do ambiente (Node.js, Expo); criação do projeto; configuração do Mosquitto; testes de conectividade MQTT | Projeto inicializado + broker MQTT funcionando |
| **2** | Telas Principais | Desenvolvimento do Dashboard e tela de Configurações; implementação da navegação entre telas; design UI/UX dark theme | Telas navegáveis com design finalizado |
| **3** | Integração MQTT | Implementação do cliente MQTT no app; conexão com broker; recepção de mensagens JSON; tratamento de reconexão automática | App recebendo alertas do PC em tempo real |
| **4** | Alarme & Notificações | Implementação da tela de alarme com som e vibração; notificações push locais; botões de confirmação/descarte; discagem de emergência | Sistema de alarme completo e funcional |
| **5** | Histórico & Persistência | Tela de histórico com armazenamento local; filtros de data; estatísticas de alertas; exportação de dados | Persistência de dados funcionando |
| **6** | Testes & Ajustes | Testes integrados com sistema completo (PC + ESP32 + App); ajustes de UX; correção de bugs; documentação final | Versão beta pronta para demonstração |

### 5.2 Marcos (Milestones)

| Marco | Data Estimada | Critério de Aceite |
|---|---|---|
| **M1 — Setup Concluído** | Fim da Semana 1 | Projeto Expo criado, broker MQTT operacional |
| **M2 — UI/UX Finalizada** | Fim da Semana 2 | Todas as telas implementadas e navegáveis |
| **M3 — MQTT Integrado** | Fim da Semana 3 | App recebe alertas do PC via MQTT em tempo real |
| **M4 — Alarme Funcional** | Fim da Semana 4 | Som, vibração e notificações funcionando |
| **M5 — Versão Beta** | Fim da Semana 6 | Sistema completo testado e documentado |

---

## 6. Passo a Passo de Implementação

### Etapa 1 — Configuração do Ambiente

```bash
# 1. Verificar Node.js (já instalado: v20.20.0 ✅)
node --version

# 2. Criar projeto Expo na pasta mobile/
cd Fall-Detect-System/mobile/
npx create-expo-app FallDetectApp --template blank

# 3. Instalar dependências do app
cd FallDetectApp
npx expo install expo-av expo-notifications expo-haptics
npm install mqtt @react-navigation/native @react-navigation/native-stack
npm install @react-native-async-storage/async-storage
npm install react-native-screens react-native-safe-area-context
```

### Etapa 2 — Configuração do Broker MQTT

```bash
# Instalar Mosquitto no PC (Ubuntu/WSL)
sudo apt update
sudo apt install mosquitto mosquitto-clients

# Habilitar e iniciar o serviço
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Configurar para aceitar conexões externas
sudo nano /etc/mosquitto/conf.d/default.conf
# Adicionar:
#   listener 1883 0.0.0.0
#   allow_anonymous true

# Reiniciar
sudo systemctl restart mosquitto

# Testar publicação (em um terminal):
mosquitto_sub -t "fall_detection/alerts" -v

# Testar assinatura (em outro terminal):
mosquitto_pub -t "fall_detection/alerts" -m '{"alert":"FALL_DETECTED","confidence":0.95,"timestamp":"2026-03-03T15:30:00"}'
```

### Etapa 3 — Estrutura do App

```
FallDetectApp/
├── App.js                    # Ponto de entrada + navegação
├── src/
│   ├── screens/
│   │   ├── DashboardScreen.js    # Tela principal
│   │   ├── AlarmScreen.js        # Tela de alarme
│   │   ├── HistoryScreen.js      # Histórico de eventos
│   │   └── SettingsScreen.js     # Configurações
│   ├── services/
│   │   ├── MqttService.js        # Conexão e gerenciamento MQTT
│   │   └── AlarmService.js       # Controle de alarme (som + vibração)
│   ├── components/
│   │   ├── StatusBadge.js        # Indicador de conexão
│   │   ├── EventCard.js          # Card de evento
│   │   └── AlarmButton.js        # Botões de ação do alarme
│   ├── storage/
│   │   └── EventStorage.js       # Persistência local
│   └── theme/
│       └── colors.js             # Paleta de cores (dark theme)
├── assets/
│   └── alarm_sound.mp3           # Som do alarme
└── app.json                      # Configuração Expo
```

### Etapa 4 — Implementação Core (MQTT + Alarme)

**Fluxo principal:**
1. App inicia → conecta ao broker MQTT
2. Assina o tópico `fall_detection/alerts`
3. Quando recebe mensagem com `alert: "FALL_DETECTED"`:
   - Navega para a tela de alarme
   - Toca alarme sonoro em volume máximo
   - Ativa vibração contínua
   - Exibe informações do alerta (confiança, timestamp)
4. Cuidador responde:
   - "Confirmar Queda" → salva evento, oferece ligar emergência
   - "Falso Alarme" → silencia, registra como falso positivo

### Etapa 5 — Testes

```bash
# 1. Iniciar o app no celular (Expo Go)
cd FallDetectApp
npx expo start

# 2. Simular uma queda via terminal
mosquitto_pub -t "fall_detection/alerts" \
  -m '{"alert":"FALL_DETECTED","confidence":0.92,"timestamp":"2026-03-03T15:30:00"}'

# 3. Testar com o script existente do projeto
python scripts/simulate_fall.py

# 4. Teste integrado completo
python scripts/main_with_esp32.py  # Detecção real + MQTT
```

---

## 7. Mudanças no Sistema Existente

### 7.1 Alterações Necessárias (Mínimas)

O sistema Python já suporta MQTT. As únicas mudanças necessárias são:

| Arquivo | Mudança |
|---|---|
| `configs/config.py` | Alterar `ESP32_CONNECTION_TYPE` para `"mqtt"` e configurar IP do broker |
| `mosquitto` | Instalar e configurar o broker MQTT no PC ou servidor |

**Nenhuma alteração** é necessária nos seguintes componentes:
- ✅ `src/model.py` — Modelo CNN+LSTM (sem mudanças)
- ✅ `src/esp32_interface.py` — Já suporta MQTT (sem mudanças)
- ✅ `hardware/esp32_fall_alert.ino` — Já suporta MQTT (só descomentar)
- ✅ `scripts/main_with_esp32.py` — Funciona com MQTT automaticamente

### 7.2 Compatibilidade

O app mobile **não substitui** o ESP32 — ambos funcionam em paralelo:
- **ESP32**: alarme físico no ambiente (para a pessoa que caiu)
- **App Mobile**: alarme remoto para o cuidador/familiar (pode estar em outro local)

---

## 8. Requisitos e Pré-requisitos

### 8.1 Requisitos de Software
- [x] Node.js 18+ (instalado: v20.20.0)
- [x] npm 10+ (instalado: 10.8.2)
- [ ] Expo CLI
- [ ] Eclipse Mosquitto (broker MQTT)
- [ ] App Expo Go no celular Android

### 8.2 Requisitos de Hardware
- PC com câmera (já existente)
- Celular Android 6.0+ com Expo Go instalado
- PC e celular na mesma rede WiFi

### 8.3 Requisitos de Rede
- Porta 1883 (MQTT) liberada no firewall do PC
- Comunicação na rede local (mesmo WiFi)

---

## 9. Riscos e Mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Latência na rede WiFi | Atraso no alarme mobile | Manter broker MQTT local; usar QoS 1 |
| App em segundo plano (Android mata processos) | Alarme não toca | Implementar notificações push via Firebase (FCM) |
| Perda de conexão MQTT | App para de receber alertas | Reconexão automática com backoff exponencial |
| Falsos positivos do modelo | Alarmes desnecessários | Limiar de confiança configurável no app |
| Firewall bloqueando MQTT | Conexão falha | Documentação de configuração de rede |

---

## 10. Próximos Passos

1. ✅ Relatório de atividades (este documento)
2. ⬜ Inicializar projeto Expo na pasta `mobile/`
3. ⬜ Configurar broker MQTT
4. ⬜ Implementar telas do app
5. ⬜ Integrar MQTT + alarme
6. ⬜ Testes integrados com o sistema completo
7. ⬜ Documentação final e demonstração

---

## 11. Referências

- **UR Fall Detection Dataset**: Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. *Computer Methods and Programs in Biomedicine*, 117(3), 489-501.
- **React Native**: https://reactnative.dev/
- **Expo**: https://expo.dev/
- **Eclipse Mosquitto**: https://mosquitto.org/
- **MQTT Protocol**: https://mqtt.org/
- **MobileNetV2**: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.

---

*Documento gerado em Março/2026 como parte do Projeto PAIC/FAPEAM — UEA*
