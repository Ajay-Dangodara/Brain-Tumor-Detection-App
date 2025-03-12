# 🔍 AI Image Classification App

## 📌 Overview

This project is an **AI-powered image classification web app** built using **Streamlit, TensorFlow, and Keras**. The app allows users to upload medical images and predicts whether the image contains a **Tumor** or is **Non-Tumor** using a pre-trained deep learning model. The UI is modern, interactive, and easy to use.

## 🚀 Features

- 🏠 **Intuitive UI** with Sidebar Navigation
- 📂 **Supports Multiple Image Formats** (`jpg`, `png`, `jpeg`)
- 🎯 **Deep Learning-based Classification (CNNs)**
- 📊 **Real-time Confidence Score & Progress Bar**
- 🔄 **Optimized Image Preprocessing** (Resizing, Normalization)
- 🌟 **Lightweight and Fast Web Application**
- 🚀 **Easy Deployment and Execution**

## 📌 Technologies Used

- **Python** - Core programming language
- **Streamlit** - Frontend UI framework for interactive web applications
- **TensorFlow & Keras** - Deep learning framework for model training and inference
- **NumPy** - For handling numerical computations and image data processing
- **Pillow (PIL)** - For image handling and resizing

## 🛠 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Ajay-Dangodara/Brain-Tumor-Detection-App.git
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Download & Place the Trained Model**
- Ensure the pre-trained model file (`model.keras`) is placed inside the `model/` directory.

## ▶️ How to Run

To launch the app, run the following command:
```bash
streamlit run app.py
```
This will start the Streamlit server and open the application in your default web browser.

## 📷 Usage Guide

Follow these simple steps to use the **AI Image Classification App**:

1️⃣ **Navigate to the Home Page**  
   - Open the application in your web browser.

2️⃣ **Upload an Image**  
   - Click on the "Choose an image..." button.  
   - Select a **JPG, PNG, or JPEG** file from your device.

3️⃣ **Click the "Classify Image" Button**  
   - The model will process the image and classify it.

4️⃣ **View the Prediction Result and Confidence Score**  
   - The predicted class (**Tumor** / **Non-Tumor**) will be displayed.  
   - A confidence percentage will indicate model certainty.

5️⃣ **Analyze the Confidence Progress Bar**  
   - A progress bar visually represents the confidence level of the prediction.

🚀 **That's it! Your image has been classified successfully.**  



## 🤖 Model Information

- **Framework Used**: TensorFlow & Keras
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Image Size**: `240x240`
- **Classification Categories**:
  - ✅ Non-Tumor
  - ❌ Tumor
- **Preprocessing Steps**:
  - Image Resizing (`240x240`)
  - Normalization (Pixel values scaled between `0-1`)
  - Batch Dimension Added (`(1, 240, 240, 3)`) for Model Compatibility


## ❓ Troubleshooting

- If you encounter a **missing dependency** issue, try reinstalling dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- If the app does not start, ensure **Streamlit** is installed:
  ```bash
  pip install streamlit
  ```
- If the **model file is missing**, download and place it inside the `model/` directory.

## 📝 License

This project is released under the **MIT License**, making it open-source and freely available for use and modification.

## 💡 Contributing

We welcome contributions! Feel free to submit **bug reports, feature requests, or pull requests** to enhance the project.

## ⭐ Show Support

If you find this project useful, consider giving it a **⭐ star on GitHub** and sharing it with your community! 🚀

---
📌 **Developed with ❤️ using Streamlit & TensorFlow**

