import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def add_extension_to_txt(file_path, extension=".jpg"):
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        updated_line = line.strip()
        if not updated_line.endswith(extension):
            updated_line += extension
        updated_lines.append(updated_line + "\n")

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    print(f"{file_path} has been updated with the extension {extension}.")


    with open(file_path, "w") as f:
        f.writelines(updated_lines)

def load_data_split(file_path, images_dir):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Te file {file_path} is not found.")

    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            image_path = os.path.join(images_dir, line.strip())
            if os.path.exists(image_path):
                data.append(image_path)
            else:
                print(f"Image not found : {image_path}")

    if not data:
        raise ValueError(f"No valid images found in{file_path}.")

    return pd.DataFrame(data, columns=["full_path"])

def prepare_labels(data_df):
    data_df["class"] = data_df["full_path"].apply(lambda x: os.path.basename(os.path.dirname(x)))
    return data_df

def get_generators(train_df, test_df, image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='full_path',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_dataframe(
        test_df,
        x_col='full_path',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator
