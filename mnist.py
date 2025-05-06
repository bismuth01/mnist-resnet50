import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, MaxPooling2D, Add, Dense, Input

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def shortcut_projection(x, filters, strides):
  projection = tf.keras.Sequential([
      Conv2D(filters, kernel_size=1, strides=strides, padding='same'),
      BatchNormalization()
  ])
  return projection(x)

def bottleneck_layers(x, kernel, strides, filters, projection_x):
  shortcut = x
  if projection_x is not None:
    shortcut = projection_x(x, filters*4, strides)

  # 1X1 REDUCING CONVOLUTION NETWORK
  x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  # KERNEL X KERNEL CONVOLUTION NETWORK
  x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  # 1X1 CONVOLUTION RESTORATION LAYER
  x = Conv2D(filters * 4, kernel_size=1, strides=1, padding='same')(x)
  x = BatchNormalization()(x)

  x = Add()([x, shortcut])
  x = ReLU()(x)

  return x

def resnet(input_shape, num_classes):
  inputs = Input(shape=input_shape)

  x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

  x = bottleneck_layers(x, kernel=3, strides=1, filters=64, projection_x=shortcut_projection)
  x = bottleneck_layers(x, kernel=3, strides=2, filters=128, projection_x=shortcut_projection)
  x = bottleneck_layers(x, kernel=3, strides=2, filters=256, projection_x=shortcut_projection)
  x = bottleneck_layers(x, kernel=3, strides=1, filters=256, projection_x=None)
  x = bottleneck_layers(x, kernel=3, strides=2, filters=512, projection_x=shortcut_projection)

  x = GlobalAveragePooling2D()(x)
  outputs = Dense(num_classes, activation='softmax')(x)

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  return model

IMG_SIZE = (28, 28)
BATCH_SIZE = 32
SEED = 42

INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 1)
NUM_CLASSES = 10
EPOCHS = 50

(training_dataset, testing_dataset), dataset_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

training_dataset = training_dataset.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
training_dataset = training_dataset.cache()
training_dataset = training_dataset.shuffle(dataset_info.splits['train'].num_examples)
training_dataset = training_dataset.batch(BATCH_SIZE)
training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)

testing_dataset = testing_dataset.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
testing_dataset = testing_dataset.batch(BATCH_SIZE)
testing_dataset = testing_dataset.cache()
testing_dataset = testing_dataset.prefetch(tf.data.AUTOTUNE)

print("Number of classes :", dataset_info.features["label"].num_classes)
print("Class labels:", dataset_info.features["label"].names)

model = resnet(INPUT_SHAPE, NUM_CLASSES)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    training_dataset,
    validation_data=testing_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

test_loss, test_accuracy = model.evaluate(testing_dataset)
print(f'Model loss: {test_loss}\nModel accuracy: {test_accuracy}')