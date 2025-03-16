import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import keras

# Image size and batch size
IMG_SIZE = (224, 224)  # ResNet requires 224x224 input
BATCH_SIZE = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(
    'Data/capsule/ground_truth',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Use 'binary' for good vs bad
)

val_generator = train_datagen.flow_from_directory(
    'Data/capsule/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Use 'binary' for good vs bad
)

# Load Pretrained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze Base Model Layers
base_model.trainable = False

# Custom Classifier Head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(6, activation='softmax')(x)  # Use Dense(1, activation='sigmoid') for binary classification
/
# Create Model
model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Unfreeze Some Layers for Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Keep most layers frozen
    layer.trainable = False

# Compile and Train Again
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save Model
model.save("capsule_resnet_classifier_v3.h5")
keras.saving.save_model(model, 'capsule_resnet_classifier_v3.keras')