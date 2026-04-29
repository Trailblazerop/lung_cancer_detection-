# 🫁 Lung Cancer Detection Using CNN
### Final Year College Project

---

## 📁 Project Structure

```
lung_cancer_project/
│
├── train_model.py          ← CNN training script
├── app.py                  ← Streamlit web application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── dataset/                ← Put your dataset here
│   ├── The IQ-OTHNCCD lung cancer dataset/
│   │   ├── Malignant cases/     (561 images)
│   │   ├── Bengin cases/        (120 images)
│   │   └── Normal cases/        (416 images)
│   └── Test cases/              (197 images)
│
├── model/                  ← Auto-created after training
│   └── lung_cancer_cnn.h5  ← Saved trained model
│
└── plots/                  ← Auto-created after training
    ├── training_history.png
    └── confusion_matrix.png
```

---

## ⚙️ How to Run the Project

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```
This will train the CNN and save the model to `model/lung_cancer_cnn.h5`

### Step 3 — Run the web app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 🧠 Full Project Explanation (For Professor Presentation)

---

### 1. What is the Problem?
Lung cancer is one of the leading causes of cancer-related deaths worldwide.
Early detection significantly improves survival rates. Manually reading CT
scans is time-consuming and depends heavily on the radiologist's expertise.
This project automates that process using Deep Learning.

---

### 2. What is a CNN?
A **Convolutional Neural Network (CNN)** is a type of deep learning model
specially designed for image analysis.

It works in layers:
- **Convolutional Layer** — detects features like edges, shapes, textures
- **Pooling Layer** — reduces image size while keeping important features
- **Fully Connected Layer** — makes the final prediction

Think of it like this:
- Layer 1 learns: edges and lines
- Layer 2 learns: shapes and curves
- Layer 3 learns: complex structures like tumors
- Final layer: classifies as Malignant / Benign / Normal

---

### 3. What is Transfer Learning?
Training a CNN from scratch needs millions of images and weeks of training.

**Transfer Learning** solves this by using a model already trained on a
large dataset (ImageNet - 14 million images) and adapting it to our task.

We used **MobileNetV2** as our base model:
- Pre-trained on 1000 categories
- We removed its top layer
- Added our own 3-class classifier on top
- Fine-tuned the last 30 layers on our lung cancer data

This gives high accuracy even with a smaller dataset.

---

### 4. Dataset Details
| Class      | Images | Meaning                        |
|------------|--------|-------------------------------|
| Malignant  | 561    | Cancerous tumor present        |
| Benign     | 120    | Non-cancerous growth           |
| Normal     | 416    | Healthy lungs, no abnormality  |
| Test Set   | 197    | Unseen images for evaluation   |

**Source:** IQ-OTH/NCCD Lung Cancer Dataset (Kaggle)

---

### 5. Data Augmentation
Since we have limited images (especially Benign: only 120), we apply
**Data Augmentation** — artificially creating more training data by:

- Rotating images slightly (±15°)
- Flipping horizontally
- Zooming in/out slightly
- Shifting left/right/up/down

This prevents **overfitting** (model memorizing training data).

---

### 6. Class Imbalance Handling
The dataset is imbalanced (Malignant: 561, Benign: 120).
Without correction, the model would be biased toward predicting Malignant.

We use **Class Weights** to give more importance to minority classes
during training, so the model learns all classes equally well.

---

### 7. Model Architecture Summary
```
Input (224×224×3 RGB Image)
       ↓
MobileNetV2 Base (pre-trained, frozen initially)
       ↓
GlobalAveragePooling2D
       ↓
BatchNormalization
       ↓
Dense(256, ReLU) + Dropout(50%)
       ↓
Dense(128, ReLU) + Dropout(30%)
       ↓
Dense(3, Softmax)  ← Final output: 3 class probabilities
```

---

### 8. Training Process
**Phase 1:** Train only top layers (base frozen) — 30 epochs
**Phase 2:** Fine-tune top 30 layers of MobileNetV2 — 15 epochs

**Loss Function:** Categorical Crossentropy
**Optimizer:** Adam (Adaptive Moment Estimation)
**Metrics:** Accuracy

**Callbacks used:**
- `ModelCheckpoint` — saves best model automatically
- `EarlyStopping` — stops training if no improvement (prevents waste)
- `ReduceLROnPlateau` — reduces learning rate when stuck

---

### 9. Evaluation Metrics
- **Accuracy** — overall correct predictions
- **Precision** — of all predicted positives, how many were correct
- **Recall** — of all actual positives, how many were caught
- **F1-Score** — balance between precision and recall
- **Confusion Matrix** — visual grid of correct vs incorrect predictions

---

### 10. Web Application (Streamlit)
The `app.py` creates an interactive web interface where:
1. Doctor/user uploads a CT scan image
2. Image is preprocessed (resized to 224×224, normalized)
3. Model predicts the class
4. Result shown with confidence percentage
5. Medical recommendation displayed

---

### 11. Technologies Used
| Technology     | Purpose                          |
|----------------|----------------------------------|
| Python         | Programming language             |
| TensorFlow/Keras | Deep learning framework        |
| MobileNetV2    | Pre-trained CNN model            |
| Streamlit      | Web application framework        |
| NumPy          | Numerical computations           |
| Pillow (PIL)   | Image loading and processing     |
| Matplotlib     | Plotting graphs                  |
| Scikit-learn   | Evaluation metrics               |

---

## 📌 Key Points to Remember for Viva

1. **Why CNN?** — CNNs automatically learn spatial features from images
2. **Why Transfer Learning?** — Limited dataset; leverage pre-trained knowledge
3. **Why MobileNetV2?** — Lightweight, fast, accurate, good for medical imaging
4. **Why Data Augmentation?** — Prevent overfitting, handle limited data
5. **Why Streamlit?** — Easy to build interactive ML web apps in Python
6. **Dataset source** — IQ-OTH/NCCD on Kaggle (real hospital CT scans)
7. **3 Classes** — Malignant (cancer), Benign (non-cancer), Normal (healthy)
