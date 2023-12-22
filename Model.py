import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
# tambaham
# Base directory of the dataset
base_dir = "dataset/"

# Image size
img_size = (224, 224)

# Batch size
batch_size = 32

# ImageDataGenerator for data with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Flow images in batches using datagen generator
images_generator = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=False
)

# Split the dataset into training and testing sets (70-30 split)
train_images, test_images, train_labels, test_labels = train_test_split(
    images_generator.filenames, images_generator.labels, test_size=0.3, random_state=42
)

# Print the number of classes
num_classes = images_generator.num_classes
print(f"Number of classes: {num_classes}")

# Create the model (same as before)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    # Adjusted for num_classes
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

# Create separate ImageDataGenerators for training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

# Train the model with validation data
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the model
model.save("model.h5")

# Evaluate the model
evaluation_result = model.evaluate(test_generator)
print(f"Test Accuracy: {evaluation_result[1] * 100:.2f}%")
