import numpy as np
from models import load_model
from tensorflow.keras.preprocessing import image
from calories_dict import calories_dict


model = load_model("../models/food101_resnet50_model.h5")
classes = list(calories_dict.keys())

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    calories = calories_dict.get(predicted_class, "Inconnu")

    print(f"Prediction : {predicted_class}, Calories : {calories} kcal")

# Test
predict_image("../data/images/apple_pie/1.jpg")
