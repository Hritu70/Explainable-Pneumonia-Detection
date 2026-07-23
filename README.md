# 🩺 Explainable Pneumonia Detection from Chest X-Rays using ResNet18 & Grad-CAM

A deep learning application that automatically classifies chest X-ray images as **Normal** or **Pneumonia** using **Transfer Learning with ResNet18** while providing **visual explanations** of every prediction through a custom implementation of **Grad-CAM**.

Unlike conventional image classification models that only output a prediction, this project emphasizes **model interpretability**, enabling users to understand **where the model is focusing** before making its decision.

---

## 📌 Project Overview

Deep learning models have demonstrated excellent performance in medical image classification. However, in healthcare applications, achieving high accuracy alone is insufficient—medical professionals also need to understand **why** a model reached a particular diagnosis.

This project addresses that challenge by combining a high-performing transfer learning model with **Grad-CAM visualizations**, allowing predictions to be interpreted rather than treated as black-box outputs.

The application enables users to:

- Upload chest X-ray images
- Predict **Normal** or **Pneumonia**
- Display prediction confidence
- Generate Grad-CAM heatmaps
- Visualize the regions influencing the model's decision

---

# 🚀 Features

- Transfer Learning using **ResNet18**
- Binary classification of Chest X-rays
- Custom **Grad-CAM** implementation from scratch
- Confidence score prediction
- Interactive Streamlit web application
- End-to-end deep learning pipeline
- Model interpretability for medical image analysis

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **90.22%** |
| Precision | **88.71%** |
| Recall | **96.67%** |
| F1 Score | **92.52%** |

### Why Recall Matters

In medical diagnosis, failing to detect a pneumonia case (**False Negative**) is considerably more serious than incorrectly predicting pneumonia for a healthy patient (**False Positive**).

To improve pneumonia detection, the model was trained using **inverse-frequency class weighting** to compensate for the dataset imbalance (approximately **2.9 : 1 Pneumonia-to-Normal ratio**), resulting in a **96.67% recall** for pneumonia cases.

---

# 🔍 Model Interpretability

To improve transparency, Grad-CAM was implemented **from scratch** using forward and backward hooks on the final convolutional layer of ResNet18.

The generated heatmaps reveal the image regions that contribute most strongly to the model's predictions.

### Observations

- Correct predictions generally focus on clinically relevant lung regions.
- Misclassified samples typically exhibit lower confidence scores than correctly classified images.
- Some incorrect predictions show attention extending toward surrounding anatomical structures, highlighting areas for future improvement.

> **Note:** Sample Grad-CAM visualizations can be added here after saving them from the application.

```
outputs/
    gradcam_normal.png
    gradcam_pneumonia.png
```

Example:

```markdown
![Grad-CAM Example](outputs/gradcam_example.png)
```

---

# 🧠 Model Architecture

- **Backbone:** ResNet18 (Pretrained on ImageNet)
- Transfer Learning
- Final Fully Connected Layer replaced for binary classification
- CrossEntropy Loss with Class Weights
- Adam Optimizer
- Best Model Checkpoint Saving
- Grad-CAM Explainability Module

---

# ⚙️ Project Workflow

### 1. Data Preparation

- Kaggle Chest X-ray Pneumonia Dataset
- Images organized using path-based DataFrames
- Stratified Train / Validation split (85:15)

### 2. Data Preprocessing

- Resize to 224 × 224
- RGB Conversion
- ImageNet Normalization
- Data Augmentation
  - Rotation
  - Affine Transform
  - Color Jitter

### 3. Model Training

- Transfer Learning using ResNet18
- Frozen feature extractor
- Fine-tuned classification layer
- Class-weighted CrossEntropy Loss

### 4. Model Evaluation

Performance evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

### 5. Explainability

Custom Grad-CAM implementation generates visual explanations highlighting the regions influencing model predictions.

### 6. Deployment

Interactive Streamlit application supporting:

- Image Upload
- Prediction
- Confidence Score
- Grad-CAM Visualization
- Heatmap Overlay

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Streamlit
- Git

---

# 📂 Project Structure

```
Explainable-Pneumonia-Detection/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── Explainable Pneumonia Detection using Grad-CAM.ipynb
│   └── Deployment.ipynb
│
├── src/
│   ├── model.py
│   ├── gradcam.py
│   └── utils.py
│
├── outputs/
│   └── training_history.csv
│
└── data_splits/
    ├── train_split.csv
    ├── val_split.csv
    └── test_split.csv
```

---

# ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/Hritu70/Explainable-Pneumonia-Detection.git
```

Navigate to the project:

```bash
cd Explainable-Pneumonia-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 📁 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

Dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Originally sourced from:

**Guangzhou Women and Children's Medical Center**

---

# 🔮 Future Improvements

- Deploy the application to the cloud.
- Train on larger and more diverse chest X-ray datasets.
- Compare Grad-CAM with Grad-CAM++.
- Extend the model to multi-class thoracic disease classification.
- Integrate uncertainty estimation for improved clinical decision support.

---

# 👨‍💻 Author

**Hrituparna Ghosh**

If you found this project interesting, feel free to ⭐ the repository.
