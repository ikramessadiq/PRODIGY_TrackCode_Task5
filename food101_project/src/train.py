from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from preprocess import add_extension_to_txt, load_data_split, get_generators
import os

# Chemins
base_dir = "../data"
images_dir = f"{base_dir}/images/images"
train_file = f"{base_dir}/meta/train.txt"
test_file = f"{base_dir}/meta/test.txt"

# Vérification des chemins
print("Chemins vérifiés :")
print(f"- Dossier images : {os.path.abspath(images_dir)}")
print(f"- Train file : {os.path.abspath(train_file)}")
print(f"- Test file : {os.path.abspath(test_file)}")

# Charger les données
train_df = add_extension_to_txt(train_file, extension=".jpg")
test_df = add_extension_to_txt(test_file, extension=".jpg")
train_df = load_data_split(train_file, images_dir)
test_df = load_data_split(test_file, images_dir)

train_generator, test_generator = get_generators(train_df, test_df)

# Construire le modèle
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compiler et entraîner
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=test_generator, epochs=10)

# Sauvegarder le modèle
os.makedirs("../models", exist_ok=True)
model.save("../models/food101_resnet50_model.h5")
print("Model trained and saved!")
