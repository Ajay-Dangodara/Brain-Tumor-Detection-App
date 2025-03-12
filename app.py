import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

IMG_WIDTH, IMG_HEIGHT = 240, 240

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("model/model.keras")

model = load_trained_model()

# Class labels
class_labels = ['Non-Tumor', 'Tumor']

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize
    img_array = np.array(img) # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2C3E50;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #7B8A8B;
        }
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            color: white;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["🏠 Home", "ℹ️ About"])

if menu == "🏠 Home":
    # Streamlit UI
    st.markdown('<p class="main-title">🔍 Brain Tumor Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload an image to classify whether it contains a Tumor or Not!</p>', unsafe_allow_html=True)

    # Upload Image
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Layout with columns for better alignment
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Display Image (Fixed Width)
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", width=250)

        # Predict Button
        if st.button("🚀 Classify Image"):
            # Preprocess and Predict
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            predictions = tf.nn.softmax(predictions).numpy()
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            # Get predicted label
            predicted_label = class_labels[predicted_class]

            # Display Results
            st.success(f"✅ Prediction: **{predicted_label}**")
            st.write(f"🎯 Confidence: **{confidence:.2%}**")

            # Show confidence bar (Fix)
            st.progress(float(np.clip(confidence, 0, 1)))  # Ensure value is in [0,1]

elif menu == "ℹ️ About":
    # About Page
    st.title("ℹ️ About")
    st.write("""
    Welcome to the **AI Image Classification App!** 🎉

    - 📌 **Model:** Trained with deep learning  
    - 📌 **Framework:** TensorFlow & Keras  
    - 📌 **Frontend:** Built with Streamlit  
      
    **How to use?**  
    1️⃣ Go to the **Home** page  
    2️⃣ Upload an image  
    3️⃣ Click on **"Classify Image"**  
    4️⃣ Get the **prediction and confidence score**  

    Try it out now! 🚀
    """)

