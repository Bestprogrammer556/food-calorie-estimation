import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================
# LOAD MODEL
# =========================================

model = tf.keras.models.load_model("current_model.h5")

# =========================================
# CLASS NAMES
# =========================================

classes = [
    "apple_pie",
    "cheesecake",
    "chicken_curry",
    "french_fries",
    "fried_rice",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "omelette",
    "pizza",
    "sushi"
]

# =========================================
# CALORIES PER 100G
# =========================================

calories_per_100g = {
    "apple_pie": 320,
    "cheesecake": 430,
    "chicken_curry": 250,
    "french_fries": 365,
    "fried_rice": 330,
    "hamburger": 295,
    "hot_dog": 290,
    "ice_cream": 207,
    "omelette": 154,
    "pizza": 266,
    "sushi": 200
}

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Food Calorie Estimator",
    page_icon="🍔",
    layout="centered"
)

# =========================================
# TITLE
# =========================================

st.title("🍕 Food Classification & Calorie Estimation")

st.write("Upload a food image and enter food weight in grams.")

# =========================================
# IMAGE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload Food Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# GRAMS INPUT
# =========================================

grams = st.number_input(
    "Enter food weight (grams)",
    min_value=1,
    max_value=2000,
    value=100
)

# =========================================
# PREDICTION
# =========================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    # Resize image
    image_resized = image.resize((160, 160))

    # Convert to array
    img_array = np.array(image_resized)

    # Preprocess
    img_array = preprocess_input(img_array)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = float(np.max(prediction) * 100)

    # Calories calculation
    calories_100g = calories_per_100g[predicted_class]

    estimated_calories = (calories_100g / 100) * grams

    # =========================================
    # RESULTS
    # =========================================

    st.subheader("Prediction Result")

    st.write(f"🍽️ Predicted Food: {predicted_class}")

    st.write(f"📊 Confidence: {confidence:.2f}%")

    st.write(f"⚖️ Weight: {grams} g")

    st.write(f"🔥 Estimated Calories: {estimated_calories:.2f} kcal")

    st.write(f"📌 Calories per 100g: {calories_100g} kcal")