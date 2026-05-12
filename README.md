# Food Image Classification & Calorie Estimation

Machine Learning final project for food image classification and calorie estimation using deep learning and transfer learning models.

---

# Project Overview

This project focuses on classifying food images into 11 categories using convolutional neural networks and transfer learning techniques.

The final system also estimates calories based on:
- predicted food category
- user-defined food weight in grams

The project includes:
- exploratory data analysis
- classical machine learning models
- custom CNN implementation
- transfer learning with MobileNetV2 and VGG16
- Streamlit web application for real-time prediction

---

# Team Members

- Ruslan Raulana
- Jexenova Inkar
- Zharikbassova Zhaniya
- Aimagambetova Karakat

SDU University  
CSS 324 — Introduction to Machine Learning  
Spring 2026

---

# Dataset

Dataset used: **Food11 Dataset**

Dataset link:  
https://www.kaggle.com/datasets/imbikramsaha/food11

### Food Categories

- apple_pie
- cheesecake
- chicken_curry
- french_fries
- fried_rice
- hamburger
- hot_dog
- ice_cream
- omelette
- pizza
- sushi

### Dataset Information

- ~9900 training images
- ~1100 testing images
- Balanced dataset
- Images resized to 224×224

---

# Data Preprocessing

The following preprocessing techniques were applied:

- image resizing
- normalization
- train/validation split
- data augmentation

### Data Augmentation

To reduce overfitting:
- rotation
- zoom
- horizontal flipping
- brightness variation

---

# Models Implemented

The following machine learning and deep learning models were compared:

| Model | Validation Accuracy |
|---|---|
| KNN | 0.2391 |
| CNN | 0.3082 |
| Logistic Regression | 0.3900 |
| VGG16 | 0.5192 |
| MobileNetV2 | **0.7884** |

---

# Final Model — MobileNetV2

MobileNetV2 achieved the best performance and was selected as the final model because of:

- high accuracy
- lightweight architecture
- fast inference
- suitability for deployment

### Training Configuration

| Parameter | Value |
|---|---|
| Input Size | 224×224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 5 / 10 |
| Loss Function | Categorical Crossentropy |

---

# Results

### Best Performing Classes

| Class | F1-score |
|---|---|
| fried_rice | 0.87 |
| chicken_curry | 0.84 |
| french_fries | 0.84 |

### Weakest Classes

| Class | F1-score |
|---|---|
| omelette | 0.60 |
| hot_dog | 0.63 |

### Common Misclassifications

Some errors occurred between visually similar foods:

- hamburger ↔ hot_dog
- pizza ↔ apple_pie
- cheesecake ↔ ice_cream

---

# Streamlit Application

An interactive Streamlit web application was developed for real-time prediction.

### Features

- upload food image
- predict food category
- display prediction confidence
- input food weight
- estimate calories

---

# Installation

Clone repository:

```bash
git clone https://github.com/your-username/food-image-classification.git
cd food-image-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app.py
```

---

# Project Structure

```bash
food-image-classification/
│
├── README.md
├── app.py
├── requirements.txt
├── report.pdf
├── best_model.h5
│
├── notebooks/
│   └── food_classification.ipynb
│
├── images/
│   ├── demo_interface.png
│   ├── confusion_matrix.png
│   └── accuracy_plot.png
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- MobileNetV2
- VGG16
- Streamlit
- NumPy
- Pandas
- Matplotlib

---

# Conclusion

- Transfer learning significantly improved performance.
- MobileNetV2 achieved the best generalization capability.
- Classical machine learning models performed worse on image data.
- Deep learning methods are highly effective for food image classification tasks.
