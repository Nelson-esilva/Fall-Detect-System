# RELATÓRIO MENSAL - PAIC/FAPEAM
**Programa de Iniciação Científica e Tecnológica da Universidade do Estado do Amazonas**

| | |
| :--- | :--- |
| **Edital:** | ( X ) Edital Nº 031/2025 - GR/UEA |
| **Tipo de Projeto:** | ( X ) PAIC/FAPEAM |
| **Título do Projeto:** | Sistema de Detecção de Quedas Utilizando Inteligência Artificial e Alerta Automatizado |
| **Aluno:** | Nelson Emeliano Silva |
| **Orientador:** | Angilberto Muniz Ferreira Sobrinho |
| **Mês de Referência:** | Mês 4 (Novembro/2025) |

---

## 1. ATIVIDADES DESENVOLVIDAS

### 1.1. Resumo das Atividades do Período
O quarto mês de execução do projeto marcou a transição fundamental da pesquisa teórica para a **implementação de engenharia de software e inteligência artificial**. As atividades concentraram-se na codificação da arquitetura de Deep Learning Híbrida (CNN + LSTM) e na construção de um pipeline robusto de processamento de dados para o dataset *UR Fall Detection*. O ambiente de desenvolvimento foi finalizado e os algoritmos de visão computacional foram implementados e testados quanto à sua integridade lógica.

### 1.2. Detalhamento Técnico e Implementação

**A. Desenvolvimento da Arquitetura Neural Híbrida (CNN-LSTM)**
Foi desenvolvido o módulo central de inteligência artificial (`src/model.py`) utilizando a biblioteca **TensorFlow/Keras**. A escolha arquitetural baseou-se na necessidade de analisar não apenas a postura estática (frame único), mas a dinâmica temporal do movimento, característica essencial para distinguir uma "queda" de um "deitar-se rapidamente".

A arquitetura implementada consiste em três blocos principais:
1.  **Extrator de Características Espaciais (CNN):** Utilizou-se a **MobileNetV2** (pré-treinada no ImageNet) com pesos congelados (*Transfer Learning*). Esta rede foi escolhida por sua baixa latência e eficiência computacional, ideal para aplicações em tempo real.
2.  **Distribuição Temporal:** O uso da camada `TimeDistributed` permite aplicar a CNN em cada um dos 20 frames da sequência de vídeo individualmente, gerando um vetor de características para cada instante de tempo.
3.  **Análise Temporal (LSTM):** Uma camada **LSTM (Long Short-Term Memory)** processa a sequência de vetores extraídos pela CNN, aprendendo a correlação entre os frames passados e atuais para identificar o padrão de queda.

*Trecho do código implementado (`src/model.py`):*
```python
def build_cnn_lstm_model():
    # Entrada: Sequência de 20 frames de 224x224 pixels (RGB)
    inputs = keras.Input(shape=(20, 224, 224, 3))

    # 1. Bloco CNN (Visão) - MobileNetV2
    base_model = keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )
    base_model.trainable = False # Congelar pesos (Transfer Learning)

    # Aplica a CNN em cada frame da sequência
    cnn_output = layers.TimeDistributed(base_model)(inputs)
    cnn_output = layers.TimeDistributed(layers.GlobalAveragePooling2D())(cnn_output)

    # 2. Bloco LSTM (Tempo)
    x = layers.LSTM(64, return_sequences=False)(cnn_output)
    x = layers.Dropout(0.5)(x) # Regularização para evitar overfitting

    # 3. Classificação
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)
```

**B. Engenharia de Dados e Pipeline ETL**
Para viabilizar o treinamento, foi necessário processar o *UR Fall Detection Dataset*, que originalmente disponibiliza sequências de imagens em arquivos comprimidos. Foi desenvolvido um script de automação (`prepare_ur_fall.py`) que realiza a Extração, Transformação e Carregamento (ETL) dos dados.

O pipeline executa as seguintes operações:
1.  **Varredura Recursiva:** Identifica automaticamente arquivos de imagem em subdiretórios profundos.
2.  **Conversão de Mídia:** Transforma sequências de imagens `.png` em arquivos de vídeo `.avi` a 30fps.
3.  **Normalização:** Redimensiona todos os inputs para a resolução nativa da rede (224x224) para evitar distorções na extração de características.

*Trecho do algoritmo de preparação de dados:*
```python
def process_extracted_folders():
    # ... Lógica de varredura de diretórios ...
    # Identificação automática de classes (Fall vs ADL/Normal)
    if "fall" in folder_name.lower():
        category = "Fall"
    elif "adl" in folder_name.lower():
        category = "Normal"
    
    # Geração do vídeo normalizado para treino
    create_video(images, output_path)
```

## 2. RESULTADOS ALCANÇADOS

Até o momento, os seguintes marcos foram atingidos:
*   **Infraestrutura de Código:** O projeto conta agora com uma base de código modular, separando claramente a lógica do modelo (`src/model.py`), o pré-processamento (`prepare_ur_fall.py`) e o loop de treinamento (`train_model.py`).
*   **Dataset Validado:** As amostras do dataset UR Fall foram processadas com sucesso, resultando em um conjunto de dados de treino organizado em classes "Fall" e "Normal", compatível com o gerador de dados do Keras.
*   **Ambiente Operacional:** As bibliotecas TensorFlow, OpenCV e NumPy foram configuradas e suas dependências solucionadas, permitindo o uso de aceleração de hardware quando disponível.

## 3. DIFICULDADES ENCONTRADAS E SOLUÇÕES TÉCNICAS

**Desafio:** Incompatibilidade de versões do TensorFlow com o ambiente Windows, gerando erros de `ModuleNotFoundError: No module named 'tensorflow.python'`.
**Análise:** O problema foi identificado como uma corrupção na instalação dos binários da biblioteca devido a limitações de caminho (MAX_PATH) e conflitos com versões prévias do Python.
**Solução:** Foi realizada uma limpeza completa do ambiente virtual (`venv`) e uma reinstalação forçada dos pacotes via `pip`, garantindo a integridade das DLLs necessárias para a execução da rede neural.

**Desafio:** Heterogeneidade dos dados brutos. O dataset original possui estruturas de pastas inconsistentes (algumas com subpastas extras, outras não).
**Solução:** Implementação de uma função de busca recursiva (`os.walk`) no script de preparação, tornando o sistema agnóstico à estrutura exata das pastas, desde que contenham os arquivos de imagem.

## 4. PRÓXIMAS ATIVIDADES (Planejamento)

Para o próximo ciclo (Mês 5), o cronograma prevê:
1.  **Execução do Treinamento:** Rodar o script `train_model.py` por múltiplos ciclos (épocas) até a convergência da função de perda (loss).
2.  **Avaliação de Métricas:** Gerar a matriz de confusão e calcular Acurácia, Precisão e Recall do modelo treinado.
3.  **Inferência em Tempo Real:** Conectar o modelo treinado (`.h5`) ao script principal (`main.py`) para realizar testes práticos utilizando a webcam do laboratório/pessoal.

---
**Assinatura do Aluno:** _________________________________________________
**Assinatura do Orientador:** _____________________________________________
