import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Base directory
base_dir = "dataset/"

# ImageDataGenerator for images
image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generator for images
train_image_generator = image_datagen.flow_from_directory(
    os.path.join(base_dir, "train/image"),
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Set to None for unsupervised learning
    shuffle=True,
    seed=42
)

test_image_generator = image_datagen.flow_from_directory(
    os.path.join(base_dir, "test/image"),
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Set to None for unsupervised learning
    shuffle=True,
    seed=42
)

# Function to load and preprocess JSON data
def load_and_preprocess_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract relevant information from JSON data
    words = [entry['text'] for line in data['valid_line'] for entry in line['words']]
    categories = [line['category'] for line in data['valid_line']]

    # Combine words and categories into a single string
    processed_data = [' '.join([word, category]) for word, category in zip(words, categories)]

    return processed_data

# Load and preprocess JSON data for train and test sets
train_json_dir = os.path.join(base_dir, "train/json")
train_json_files = os.listdir(train_json_dir)
processed_train_json_data = [load_and_preprocess_json(os.path.join(train_json_dir, file)) for file in train_json_files]

test_json_dir = os.path.join(base_dir, "test/json")
test_json_files = os.listdir(test_json_dir)
processed_test_json_data = [load_and_preprocess_json(os.path.join(test_json_dir, file)) for file in test_json_files]

# Convert the data into tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'image_input': train_image_generator[0][0], 'json_input': processed_train_json_data}
)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {'image_input': test_image_generator[0][0], 'json_input': processed_test_json_data}
)).batch(32)

# Model for image data
image_input = Input(shape=(224, 224, 3), name='image_input')
flattened_image = Flatten()(image_input)
dense_image = Dense(128, activation='relu')(flattened_image)

# Model for text data
json_input = Input(shape=(1,), dtype=tf.string, name='json_input')
vectorize_layer = preprocessing.TextVectorization(
    max_tokens=500,  # Adjust based on your vocabulary size
    output_mode='int',
    output_sequence_length=1  # Set to 1 since we're treating each document as a single token
)
processed_json = vectorize_layer(json_input)
processed_json = tf.cast(processed_json, tf.float32)  # Cast to float32
dense_json = Dense(128, activation='relu')(processed_json)

# Combine the two models
merged = concatenate([dense_image, dense_json])

# Output layer for clustering task
output_layer = Dense(3, activation='softmax', name='output')(merged)

model = Model(inputs=[image_input, json_input], outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='kld', metrics=['accuracy'])  # Kullback-Leibler Divergence as a loss function

# Train the model
model.fit(train_dataset, epochs=10, validation_split=0.1)

# Save the model
model.save("unsupervised_model.h5")

# Evaluate the model (in unsupervised learning, there's no traditional evaluation, but you can evaluate how well the model clusters)
