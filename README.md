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
├── src/
│   └── model.py              # Arquitetura CNN-LSTM
├── data/
│   └── raw/
│       ├── Fall/             # Vídeos de quedas
│       └── Normal/           # Vídeos de atividades normais (ADL)
├── models/                   # Modelos treinados (.h5)
├── docs/                     # Relatórios e artigos de referência
├── UR_Fall_Downloads/        # Dados brutos do dataset (temporário)
├── collect_videos.py         # Coletor de dados via webcam
├── prepare_ur_fall.py        # Processador do dataset UR Fall
├── train_model.py            # Script de treinamento
├── main.py                   # Detecção em tempo real
└── requirements.txt          # Dependências
```

---

## 🚀 Como Usar

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
python prepare_ur_fall.py
```

**Opção B: Gravar seus próprios dados**
```bash
python collect_videos.py
```
- Pressione `f` para gravar quedas
- Pressione `n` para gravar atividades normais

### 3. Treinar o Modelo
```bash
python train_model.py
```

### 4. Executar Detecção em Tempo Real
```bash
python main.py
```

---

## 📊 Dataset Utilizado

**UR Fall Detection Dataset**
- 30 sequências de quedas
- 40 sequências de atividades diárias (ADL)
- Câmeras RGB + Profundidade + Acelerômetro

> Referência: Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. *Computer Methods and Programs in Biomedicine*, 117(3), 489-501.

---

## 👨‍💻 Autor

**Nelson Emeliano Silva**  
Orientador: Prof. Angilberto Muniz Ferreira Sobrinho  
Universidade do Estado do Amazonas - UEA

---

## 📄 Licença

Este projeto é para fins acadêmicos e de pesquisa.
