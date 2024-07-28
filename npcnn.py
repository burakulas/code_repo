"""
Author: Burak Ulas -  github.com/burakulas
2024, Konkoly Observatory, COMU

Image and annotation data at the following link are suitable for use in this model:
https://drive.google.com/file/d/1rOWSGWYOVPwSEaB2TUI1NzIVsWGU9T3T/view?usp=sharing
"""

import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
import absl.logging
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

np.random.seed(37)
random.seed(1254)
tf.random.set_seed(89)

# Select specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()


input_size = 240 # image size (240x240)
n_class = 2 # Number of classes ('pul' and 'min')
EPOCHS = 100 # Number of epochs

# Train and validation data
tr_folder = "/path/to/train/"
val_folder = "/path/to/validation/"

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomTranslation(0.05, 0.05),
    layers.experimental.preprocessing.RandomContrast(0.1)
])


# Build the feature extractor
def feature_extract(inputs):
    x = tf.keras.layers.Conv2D(16, kernel_size=3, input_shape=(input_size, input_size, 1),kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    return x


# Process feture maps
def dense_headf(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Dropout 
    return x


# Classification
def classifier_head(inputs):
    return tf.keras.layers.Dense(n_class, activation='softmax', name='classifier_head')(inputs)


# Regression
def regressor_headf(inputs):
    return tf.keras.layers.Dense(units=8, name='regressor_head')(inputs)


# Confidence
def confidence_headf(inputs):
    return tf.keras.layers.Dense(units=1, activation='sigmoid', name='confidence_head')(inputs)
    

# Build complete model
def build_model(inputs):
    feature_extractor = feature_extract(inputs)
    dense_head = dense_headf(feature_extractor)
    classification_head = classifier_head(dense_head)
    regressor_head = regressor_headf(dense_head)  # Output shape will be (None, 8)
    regressor_head_reshaped = tf.reshape(regressor_head, (-1, 2, 4))  # Reshape to (batch_size, 2, 4)
    confidence_head = confidence_headf(dense_head)
    
    model = tf.keras.Model(inputs=inputs, outputs=[classification_head, regressor_head_reshaped, confidence_head], name='object_detector')
    return model


# Arrange files
def data_files(train_path=tr_folder, val_path=val_folder, image_ext='.png', ):
    train_image_files = []
    train_annotation_files = []
    train_image_paths = []
    train_annotation_paths = []
    val_image_files = []
    val_annotation_files = []
    val_image_paths = []
    val_annotation_paths = []
    for r, d, f in os.walk(train_path):
        for file in f:
            if file.endswith(image_ext):
                train_image_files.append(os.path.join(r, file))
                train_annotation_files.append(os.path.join(r, file.replace(image_ext, '.txt')))
                train_image_paths.append(os.path.join(train_path, os.path.splitext(file)[0] + '.png'))
                train_annotation_paths.append(os.path.join(train_path, os.path.splitext(file)[0] + '.txt'))
    
    for r, d, f in os.walk(val_path):
        for file in f:
            if file.endswith(image_ext):
                val_image_files.append(os.path.join(r, file))
                val_annotation_files.append(os.path.join(r, file.replace(image_ext, '.txt')))
                val_image_paths.append(os.path.join(val_path, os.path.splitext(file)[0] + '.png'))
                val_annotation_paths.append(os.path.join(val_path, os.path.splitext(file)[0] + '.txt'))
   
    trsize = len(train_image_files)
    valsize = len(val_image_files)
    trindices = np.arange(trsize)
    valindices = np.arange(valsize)
    
    train_images = [train_image_paths[i] for i in trindices]
    train_annotations = [train_annotation_paths[i] for i in trindices]
    val_images = [val_image_paths[i] for i in valindices]
    val_annotations = [val_annotation_paths[i] for i in valindices]
    
    return (train_images, train_annotations), (val_images, val_annotations)
    

(train_image_files, train_annotation_files), (val_image_files, val_annotation_files) = data_files()

print("----------- Number of images -----------")
print(f"Training: {len(train_image_files)} images, {len(train_annotation_files)} annotations")
print(f"Validation: {len(val_image_files)} images, {len(val_annotation_files)} annotations")


# Data preparation
def load_and_preprocess_data(image_path, annotation_path, aug, image_size=(240, 240)):
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Load as grayscale image
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    
    #Augmentation only on train set
    if aug == 1:
        image = tf.expand_dims(image, 0)  # Add batch dimension
        image = data_augmentation(image)
        image = tf.squeeze(image, 0)  # Remove batch dimension
        
    if aug == 0:
        print(" ")

    # Read and preprocess the annotation
    annotation = tf.io.read_file(annotation_path)
    annotation = tf.strings.strip(annotation)
    annotation_lines = tf.strings.split(annotation, '\n')  # Remove last empty element

    annotations = tf.strings.split(annotation_lines, ' ')
    annotations = tf.strings.to_number(annotations, tf.float32)

    # Keep the shape (N, 5)
    annotations = tf.reshape(annotations, [-1, 5])

    # Separate class labels and bounding boxes
    class_labels = annotations[:, 0]  # Assuming first element is class label
    bounding_boxes = annotations[:, 1:]
    confidence_scores = tf.ones_like(class_labels, dtype=tf.float32)  # Replace with your own logic
    
    # use for debugging
    #print("----------------------")
    #tf.print("impath:", image_path, "anpath:",annotation_path,"Class labels:", class_labels, "Annotations:", annotations, "Bounding boxes:", bounding_boxes, "conf:",confidence_scores)
    
    return image, (class_labels, bounding_boxes, confidence_scores)
    

# Dataset creation
def create_dataset(image_files, annotation_files, aug, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((image_files, annotation_files))
    dataset = dataset.map(lambda img_path, ann_path: load_and_preprocess_data(img_path, ann_path, aug), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Define loss
def smooth_l1_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    abs_error = K.abs(y_true - y_pred)
    delta = 1.0
    quadratic_part = K.clip(abs_error, 0.0, delta)
    linear_part = abs_error - quadratic_part
    loss = 0.5 * K.square(quadratic_part) + delta * linear_part
    return K.sum(loss, axis=-1)

train_dataset = create_dataset(train_image_files, train_annotation_files, 1, batch_size=64)
val_dataset = create_dataset(val_image_files, val_annotation_files, 0, batch_size=64)


# Use for debugging
#def inspect_dataset(dataset):
#    for image, (class_labels, bounding_boxes) in dataset.take(1):
#        print("Image shape:", image.shape)
#        print("Class labels:", class_labels.numpy())
#        print("Bounding boxes:", bounding_boxes.numpy())  

#for image, (class_labels, bounding_boxes) in train_dataset.take(1):
#    print("Image shape:", image.shape)
#    print("Class labels shape:", class_labels.shape)
#    print("Bounding boxes shape:", bounding_boxes.shape)
#inspect_dataset(train_dataset)
#inspect_dataset(val_dataset)


# Model input
inputs = tf.keras.layers.Input(shape=(input_size, input_size, 1))


# Build the model
model = build_model(inputs)
model.summary()

# Define learning rate scheduler
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   
    factor=0.1,           
    patience=10,          
    verbose=1,            
    mode='auto',          
    min_lr=1e-6           
)


# Optimizer
opt = keras.optimizers.Adam(learning_rate=0.0005)


# Model compilation (Note that tf.reshape is box regressor)
model.compile(optimizer=opt,
              loss={'classifier_head': 'binary_crossentropy', 'tf.reshape': smooth_l1_loss,'confidence_head': 'mean_squared_error'},
              metrics={'classifier_head': 'accuracy', 'tf.reshape': 'accuracy', 'confidence_head': 'accuracy'})



csv_logger = CSVLogger('training.log') # write log
history_logger = tf.keras.callbacks.CSVLogger('hist.csv', separator=",", append=True) # Write history
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # Set early stopping
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Create log directory
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) # Create tensorboard callback
mc = ModelCheckpoint(filepath='{epoch}.h5', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=1) # Set model checkpoint


# Train the model
model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[history_logger,tensorboard_callback,csv_logger,reduce_lr,es,mc], verbose=1)
