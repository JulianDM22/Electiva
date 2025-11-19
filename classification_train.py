# classification_train.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# CONFIG
DATA_DIR = "data/tomatoes_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10  # puedes subir a 20 si tu GPU aguanta

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "valid")
test_dir  = os.path.join(DATA_DIR, "test")

# 1) Cargar datasets
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Clases:", class_names)

# Prefetch para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# 2) Función genérica para crear modelo con un backbone de Keras Applications
def build_model(base_model_fn, preprocess_input, input_shape=IMG_SIZE + (3,)):
    base_model = base_model_fn(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # primero congelamos

    inputs = keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_evaluate(model, name):
    print(f"\nEntrenando modelo: {name}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print(f"\nEvaluando modelo: {name}")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy ({name}):", test_acc)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{name}_tomatoes.h5")
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

    return model, history

# 3) Entrenar ResNet50
from tensorflow.keras.applications import resnet
resnet_model = build_model(
    keras.applications.ResNet50,
    resnet.preprocess_input
)
resnet_model, resnet_history = train_and_evaluate(resnet_model, "resnet50")

# 4) Entrenar EfficientNetB0
from tensorflow.keras.applications import efficientnet
effnet_model = build_model(
    keras.applications.EfficientNetB0,
    efficientnet.preprocess_input
)
effnet_model, effnet_history = train_and_evaluate(effnet_model, "efficientnetb0")

# 5) Entrenar DenseNet121
from tensorflow.keras.applications import densenet
densenet_model = build_model(
    keras.applications.DenseNet121,
    densenet.preprocess_input
)
densenet_model, densenet_history = train_and_evaluate(densenet_model, "densenet121")

# 6) Ejemplo de predicción unitaria
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_single_image(img_path, model, preprocess_input, model_name):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = tf.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    pred_idx = np.argmax(preds[0])
    print(f"Modelo {model_name} predice: {class_names[pred_idx]} (prob={preds[0][pred_idx]:.4f})")

# Prueba con alguna imagen del test
sample_img = os.path.join(test_dir, class_names[0],
                        os.listdir(os.path.join(test_dir, class_names[0]))[0])

predict_single_image(sample_img, resnet_model, resnet.preprocess_input, "resnet50")
predict_single_image(sample_img, effnet_model, efficientnet.preprocess_input, "efficientnetb0")
predict_single_image(sample_img, densenet_model, densenet.preprocess_input, "densenet121")
