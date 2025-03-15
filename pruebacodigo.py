import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Cambiar el directorio de trabajo al del script (si es necesario)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. Preparación de datos (simplificada)
# ============================================================================
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, "tiny-imagenet-200", "train")
val_img_dir = os.path.join(base_dir, "tiny-imagenet-200", "val", "images")
test_img_dir = os.path.join(base_dir, "tiny-imagenet-200", "test", "images")

# Leer los CSVs sin modificar la columna Encoded_Label
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")
test_df = pd.read_csv("test_data.csv")

# Para train: limpiar la ruta y construir el path completo
train_df['File'] = train_df['File'].str.replace("/kaggle/input/tiny-imagenet", "", regex=False).str.lstrip("/")
train_df['FullPath'] = train_df['File'].apply(lambda x: os.path.join(base_dir, x))

# Para val y test: construir el path completo a partir del nombre del archivo
val_df['FullPath'] = val_df['File'].str.strip().apply(lambda x: os.path.join(val_img_dir, x))
test_df['FullPath'] = test_df['File'].str.strip().apply(lambda x: os.path.join(val_img_dir, x))

# ============================================================================
# 2. Data Augmentation y Generadores
# ============================================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='FullPath',
    y_col='Encoded_Label',
    target_size=(64, 64),
    class_mode='raw',  # Se mantiene para trabajar con etiquetas numéricas
    batch_size=64,
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='FullPath',
    y_col='Encoded_Label',
    target_size=(64, 64),
    class_mode='raw',
    batch_size=64,
    shuffle=False
)

test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='FullPath',
    y_col=None,
    class_mode=None,
    target_size=(64, 64),  # Ajustar según lo que requiera el modelo
    batch_size=32,
    shuffle=False
)

# ============================================================================
# 3. Modelo base pre-entrenado (EfficientNetB0) y Configuración
# ============================================================================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(200)(x)  # Sin activación, ya que se usará from_logits=True
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ============================================================================
# 4. Callbacks y Entrenamiento
# ============================================================================
early_stop = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

EPOCHS = 25
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ============================================================================
# 7. Fine-tuning 
# ============================================================================
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=EPOCHS,           
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ============================================================================
# 8. Carga del mejor modelo
# ============================================================================
model.load_weights('best_model.h5')

# ============================================================================
# 9. Predicciones en test
# ============================================================================
test_generator.reset()
pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)

# ============================================================================
# 10. CSV final
# ============================================================================
submission = pd.DataFrame({
    'id': test_df['File'],
    'pred': pred_classes
})
submission.to_csv('submission.csv', index=False)
print("¡CSV de predicciones generado con éxito, Papi!")