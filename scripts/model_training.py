import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained model
pretrained_model = EfficientNetB7(include_top=False, weights="imagenet", pooling="max", input_shape=(224, 224, 3))
pretrained_model.trainable = False

# Model architecture setup
inputs = tf.keras.Input(shape=(224, 224, 3))
x = pretrained_model(inputs, training=False)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)
outputs = Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer=Adam(0.00001), loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
checkpoint_filepath = 'models/checkpoints/model-{epoch:02d}-{val_accuracy:.2f}.keras'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

# Data augmentation and data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_images = train_datagen.flow_from_directory(
    "datasets/animals10/raw-img/",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    seed=42
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_images = validation_datagen.flow_from_directory(
    "datasets/animals10/raw-img/",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    seed=42
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=100,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr],
)
