# 🚨 Sistema de Detecção de Quedas com IA

Sistema de detecção de quedas em tempo real utilizando **Deep Learning (CNN + LSTM)** e visão computacional.

**Projeto de Iniciação Científica - PAIC/FAPEAM - Universidade do Estado do Amazonas (UEA)**

---

## 📋 Sobre o Projeto

Este sistema utiliza uma arquitetura híbrida de Redes Neurais para detectar quedas através de câmeras de vídeo:

1. **MobileNetV2 (CNN):** Extrai características visuais de cada frame.
2. **LSTM:** Analisa a sequência temporal de 20 frames para identificar o padrão de movimento de queda.

---

## 📁 Estrutura do Projeto

```
Fall-Detect-System/
├── src/                          # Código fonte principal
│   ├── model.py                  # Arquitetura CNN-LSTM
│   └── esp32_interface.py       # Interface de comunicação com ESP32
│
├── scripts/                      # Scripts executáveis
│   ├── main.py                   # Detecção em tempo real (básico)
│   ├── main_with_esp32.py        # Detecção com integração ESP32
│   ├── train_model.py            # Treinamento do modelo
│   ├── prepare_ur_fall.py        # Processamento do dataset UR Fall
│   ├── collect_videos.py          # Coletor de dados via webcam
│   ├── simulate_fall.py          # Simulação de detecção de quedas
│   ├── generate_example_logs.py  # Geração de logs de exemplo
│   └── generate_topic_images.py  # Geração de diagramas
│
├── hardware/                     # Código para hardware
│   └── esp32_fall_alert.ino      # Código Arduino para ESP32
│
├── tests/                        # Testes
│   └── test_esp32.py             # Teste de conexão ESP32
│
├── configs/                      # Configurações
│   └── config.py                 # Configurações centralizadas
│
├── docs/                         # Documentação
│   ├── ESP32_INTEGRATION.md      # Guia de integração ESP32
│   ├── RESULTADOS_OBTIDOS.txt    # Resultados do projeto
│   └── *.pdf                     # Relatórios e artigos
│
├── assets/                       # Recursos visuais
│   ├── otimizacao_modelo.png
│   ├── protocolos_alerta_fcm.png
│   └── documentacao_tecnica.png
│
├── data/                         # Dados
│   └── raw/                      # Dados brutos
│       ├── Fall/                 # Vídeos de quedas
│       └── Normal/                # Vídeos de atividades normais (ADL)
│
├── models/                       # Modelos treinados (.h5)
│
├── logs/                         # Logs de treinamento
│   └── run-YYYYMMDD-HHMMSS/      # Logs de cada execução
│
├── UR_Fall_Downloads/           # Dados brutos do dataset (temporário)
│
├── requirements.txt              # Dependências Python
├── .gitignore                    # Arquivos ignorados pelo Git
└── README.md                     # Este arquivo
```

---

## 🚀 Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Preparar o Dataset

**Opção A: Usar o UR Fall Detection Dataset (Recomendado)**
1. Baixe os arquivos `*-cam0-rgb.zip` (Falls e ADLs) de: https://fenix.ur.edu.pl/mkepski/ds/uf.html
2. Extraia na pasta `UR_Fall_Downloads/`
3. Execute:
```bash
python scripts/prepare_ur_fall.py
```

**Opção B: Gravar seus próprios dados**
```bash
python scripts/collect_videos.py
```
- Pressione `f` para gravar quedas
- Pressione `n` para gravar atividades normais

### 3. Treinar o Modelo

```bash
python scripts/train_model.py
```

Os logs de performance serão salvos automaticamente em `logs/run-YYYYMMDD-HHMMSS/`.

### 4. Executar Detecção em Tempo Real

**Versão básica (sem ESP32):**
```bash
python scripts/main.py
```

**Versão com integração ESP32:**
```bash
python scripts/main_with_esp32.py
```

---

## 🔌 Integração com ESP32

O sistema pode enviar alertas para um ESP32 que dispara alarmes locais (buzzer, LEDs) quando uma queda é detectada.

### Configuração Rápida

1. **Carregue o código no ESP32:**
   - Abra `hardware/esp32_fall_alert.ino` no Arduino IDE
   - Instale a biblioteca ArduinoJson
   - Faça upload para o ESP32

2. **Configure a porta no Python:**
   - Edite `configs/config.py` e ajuste `ESP32_PORT` (ex: "COM3")

3. **Teste a conexão:**
   ```bash
   python tests/test_esp32.py
   ```

4. **Execute o sistema completo:**
   ```bash
   python scripts/main_with_esp32.py
   ```

📖 **Documentação completa:** Veja `docs/ESP32_INTEGRATION.md`

---

## 📊 Dataset Utilizado

**UR Fall Detection Dataset**
- 30 sequências de quedas
- 40 sequências de atividades diárias (ADL)
- Câmeras RGB + Profundidade + Acelerômetro

> Referência: Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. *Computer Methods and Programs in Biomedicine*, 117(3), 489-501.

---

## 🧪 Scripts Auxiliares

### Simulação de Quedas
```bash
python scripts/simulate_fall.py
```
Demonstra o sistema processando vídeos de quedas do dataset.

### Gerar Logs de Exemplo
```bash
python scripts/generate_example_logs.py
```
Gera logs de exemplo sem precisar treinar o modelo.

---

## 📈 Logs de Performance

Cada execução de treinamento cria uma pasta em `logs/run-YYYYMMDD-HHMMSS/` com:
- `run_metadata.json` - Configurações do treinamento
- `history.json` / `history.csv` - Histórico de treinamento
- `curves.png` - Gráficos de loss e accuracy
- `confusion_matrix.png` - Matriz de confusão
- `classification_report.txt` - Relatório de classificação
- E mais...

Veja `logs/README.md` para mais detalhes.

---

## ⚙️ Configuração

As configurações principais estão em `configs/config.py`:
- Caminhos de diretórios
- Parâmetros do modelo
- Configurações ESP32
- Hiperparâmetros

---

## 👨‍💻 Autor

**Nelson Emeliano Silva**  
Orientador: Prof. Angilberto Muniz Ferreira Sobrinho  
Universidade do Estado do Amazonas - UEA

---

## 📄 Licença

Este projeto é para fins acadêmicos e de pesquisa.

---

## 🔗 Links Úteis

- [Documentação ESP32](docs/ESP32_INTEGRATION.md)
- [UR Fall Dataset](https://fenix.ur.edu.pl/mkepski/ds/uf.html)
- [Logs de Performance](logs/README.md)
