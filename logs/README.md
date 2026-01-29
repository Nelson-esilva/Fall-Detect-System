# Logs de Performance - Sistema de Detecção de Quedas

Esta pasta contém os logs de performance de cada execução de treinamento do modelo.

## Estrutura

Cada execução de treinamento cria uma pasta `run-YYYYMMDD-HHMMSS/` com os seguintes arquivos:

### Arquivos de Metadados

- **`run_metadata.json`**: Configurações do treinamento (hiperparâmetros, dataset, etc.)
- **`final_metrics.json`**: Métricas finais (loss e accuracy no conjunto de teste)

### Histórico de Treinamento

- **`history.json`**: Histórico completo em formato JSON (loss, accuracy, val_loss, val_accuracy por época)
- **`history.csv`**: Mesmo histórico em formato CSV para análise em planilhas
- **`keras_history.csv`**: Log do CSVLogger do Keras
- **`curves.png`**: Gráficos de loss e accuracy ao longo das épocas

### Avaliação do Modelo

- **`classification_report.txt`**: Relatório de classificação com precisão, recall e F1-score por classe
- **`confusion_matrix.json`**: Matriz de confusão em formato JSON
- **`confusion_matrix.png`**: Visualização da matriz de confusão

### Dados para Reprodução (opcional)

- **`y_test.npy`**: Labels verdadeiros do conjunto de teste
- **`y_pred.npy`**: Predições do modelo
- **`y_prob.npy`**: Probabilidades preditas

## Como Usar

### Gerar Logs Reais

Execute o treinamento normalmente:

```bash
python train_model.py
```

Os logs serão automaticamente salvos em `logs/run-YYYYMMDD-HHMMSS/`.

### Gerar Logs de Exemplo

Para ver a estrutura dos logs sem treinar o modelo:

```bash
python generate_example_logs.py
```

## Versionamento

**Recomendação**: Versionar apenas arquivos leves (JSON, CSV, TXT, PNG). Os arquivos `.npy` são grandes e podem ser ignorados.

Atualize o `.gitignore` se necessário:

```
# Logs - versionar apenas arquivos leves
logs/**/*.npy
logs/**/*.h5
```

## Análise dos Logs

### Comparar Execuções

1. Compare `final_metrics.json` de diferentes runs
2. Visualize `curves.png` para ver a evolução do treinamento
3. Analise `confusion_matrix.png` para entender os erros do modelo

### Análise em Python

```python
import json
import pandas as pd

# Carregar histórico
with open('logs/run-20260129-095557/history.json') as f:
    history = json.load(f)

# Converter para DataFrame
df = pd.DataFrame(history)
print(df.tail())  # Últimas épocas
```

## Estrutura de Diretórios

```
logs/
├── .gitkeep
├── README.md
└── run-YYYYMMDD-HHMMSS/
    ├── run_metadata.json
    ├── history.json
    ├── history.csv
    ├── keras_history.csv
    ├── curves.png
    ├── final_metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.json
    ├── confusion_matrix.png
    ├── y_test.npy (opcional)
    ├── y_pred.npy (opcional)
    └── y_prob.npy (opcional)
```
