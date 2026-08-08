import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm_model(
    sequence_length: int = 20,
    img_height: int = 128,
    img_width: int = 128,
    fine_tune_layers: int = 0,
    learning_rate: float = 1e-4,
    lstm_units: int = 64,
):
    """
    Arquitetura híbrida CNN-LSTM para detecção de quedas.

    Args:
        sequence_length: frames por janela temporal
        img_height, img_width: resolução de entrada
        fine_tune_layers: quantas camadas finais da MobileNetV2 descongelar
                          (0 = transfer learning puro, 30 = bom equilíbrio)
        learning_rate: taxa de aprendizado (usar valor menor quando fine-tuning)
        lstm_units: número de unidades na camada LSTM
    """
    input_shape = (sequence_length, img_height, img_width, 3)
    inputs = keras.Input(shape=input_shape)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
    )

    base_model.trainable = False
    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    cnn_output = layers.TimeDistributed(base_model)(inputs)
    cnn_output = layers.TimeDistributed(layers.GlobalAveragePooling2D())(cnn_output)

    x = layers.LSTM(lstm_units, return_sequences=False)(cnn_output)
    x = layers.Dropout(0.5)(x)
    # Cast to float32 before sigmoid — required for mixed precision stability
    x = layers.Activation('linear', dtype='float32')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def build_cnn_only_model(
    sequence_length: int = 20,
    img_height: int = 128,
    img_width: int = 128,
    fine_tune_layers: int = 0,
    learning_rate: float = 1e-4,
):
    """
    Baseline CNN-only: processa cada frame independentemente com MobileNetV2,
    faz pooling temporal (média) e classifica. Sem componente temporal (LSTM).
    Serve para demonstrar o valor da modelagem temporal.
    """
    input_shape = (sequence_length, img_height, img_width, 3)
    inputs = keras.Input(shape=input_shape)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
    )

    base_model.trainable = False
    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    cnn_output = layers.TimeDistributed(base_model)(inputs)
    cnn_output = layers.TimeDistributed(layers.GlobalAveragePooling2D())(cnn_output)

    x = layers.GlobalAveragePooling1D()(cnn_output)
    x = layers.Dropout(0.5)(x)
    x = layers.Activation('linear', dtype='float32')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def convert_to_tflite(model, output_path: str, quantize: bool = True):
    """
    Converte modelo Keras para TFLite com quantização opcional.
    Quantização INT8 reduz o tamanho ~4x e melhora velocidade em CPU.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite salvo em {output_path} ({size_mb:.1f} MB)")
    return output_path
