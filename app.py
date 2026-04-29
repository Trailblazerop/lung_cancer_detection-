"""
=============================================================
  LUNG CANCER DETECTION - STREAMLIT WEB APPLICATION
  Final Year Project
=============================================================
  Run with: streamlit run app.py
=============================================================
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        color: #2c3e50;
    }

    .main-header {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #3d72e8 0%, #6997ff 48%, #94b9ff 100%);
        padding: 4rem 2rem 3rem;
        border-radius: 32px;
        margin-bottom: 2.4rem;
        box-shadow: 0 20px 60px rgba(58, 90, 204, 0.18);
        border: 1px solid rgba(255,255,255,0.32);
    }

    .main-header::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 10% 20%, rgba(255,255,255,0.24), transparent 18%),
                    radial-gradient(circle at 85% 20%, rgba(255,255,255,0.18), transparent 18%),
                    radial-gradient(circle at 50% 80%, rgba(255,255,255,0.14), transparent 20%);
        pointer-events: none;
    }

    .hero-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        max-width: 1180px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }

    .hero-illustration {
        flex: 0 0 48%;
        min-width: 280px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .hero-illustration svg {
        width: 100%;
        height: auto;
    }

    .hero-content {
        flex: 1;
        text-align: left;
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.22);
        backdrop-filter: blur(12px);
        color: #f9fbff;
        padding: 0.85rem 1.4rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255,255,255,0.22);
        margin-bottom: 1.2rem;
    }

    .hero-content h1 {
        color: #f8fbff;
        font-size: 3.4rem;
        line-height: 1.02;
        margin: 0;
        letter-spacing: -1px;
        font-weight: 800;
        text-shadow: 0 24px 60px rgba(14, 24, 71, 0.18);
    }

    .hero-copy {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.96);
        margin-bottom: 1.7rem;
        max-width: 640px;
        line-height: 1.75;
    }

    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1.6rem;
    }

    .hero-stat {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 22px;
        padding: 1rem 1.2rem;
        min-height: 120px;
    }

    .hero-stat strong {
        display: block;
        color: #ffffff;
        font-size: 1.55rem;
        margin-bottom: 0.35rem;
    }

    .hero-stat span {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
    }

    .hero-accent {
        position: absolute;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
        filter: blur(46px);
        top: -90px;
        right: -100px;
        z-index: 0;
    }

    .hero-accent-small {
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: rgba(255,255,255,0.06);
        filter: blur(22px);
        bottom: -60px;
        left: -50px;
        z-index: 0;
    }

    @media (max-width: 900px) {
        .hero-card {
            flex-direction: column;
            align-items: center;
        }
        .hero-content {
            text-align: center;
        }
        .hero-content h1 {
            font-size: 2.8rem;
        }
        .hero-copy {
            margin-left: 0;
        }
        .main-header {
            padding: 3rem 1.3rem 2.5rem;
        }
        .hero-illustration {
            width: 90%;
        }
        .hero-stats {
            grid-template-columns: 1fr;
        }
    }

    .result-malignant {
        background: linear-gradient(135deg, #ff5858 0%, #ff4444 50%, #ff2b2b 100%);
        color: white;
        padding: 2.2rem 2.8rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 1.8rem 0;
        box-shadow: 0 16px 45px rgba(255, 68, 68, 0.32);
        border: 1px solid rgba(255,255,255,0.22);
        transition: all 0.3s ease;
        letter-spacing: -0.5px;
    }

    .result-malignant:hover {
        transform: translateY(-6px);
        box-shadow: 0 28px 60px rgba(255, 59, 74, 0.35);
    }

    .result-benign {
        background: linear-gradient(135deg, #ffb84d 0%, #ffa936 50%, #ff9f1a 100%);
        color: white;
        padding: 2.2rem 2.8rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 1.8rem 0;
        box-shadow: 0 16px 45px rgba(255, 169, 54, 0.32);
        border: 1px solid rgba(255,255,255,0.22);
        transition: all 0.3s ease;
        letter-spacing: -0.5px;
    }

    .result-benign:hover {
        transform: translateY(-6px);
        box-shadow: 0 28px 60px rgba(243, 156, 18, 0.35);
    }

    .result-normal {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 50%, #16a34a 100%);
        color: white;
        padding: 2.2rem 2.8rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 1.8rem 0;
        box-shadow: 0 16px 45px rgba(34, 197, 94, 0.32);
        border: 1px solid rgba(255,255,255,0.22);
        transition: all 0.3s ease;
        letter-spacing: -0.5px;
    }

    .result-normal:hover {
        transform: translateY(-6px);
        box-shadow: 0 28px 60px rgba(39, 174, 96, 0.35);
    }

    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, rgba(250, 252, 255, 0.99) 100%);
        border: 1.5px solid rgba(94, 125, 255, 0.22);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 35px rgba(94, 125, 255, 0.12);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .info-card:hover {
        box-shadow: 0 14px 45px rgba(94, 125, 255, 0.16);
        border-color: rgba(94, 125, 255, 0.35);
    }

    .confidence-bar-container {
        background: rgba(94, 125, 255, 0.15);
        border-radius: 50px;
        height: 20px;
        margin: 10px 0 18px 0;
        overflow: hidden;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.04);
        border: 1px solid rgba(94, 125, 255, 0.18);
    }

    .stButton > button {
        background: linear-gradient(135deg, #5e7dff 0%, #7aa3ff 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 16px;
        padding: 1rem 2.8rem;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.35s ease;
        box-shadow: 0 8px 28px rgba(94, 125, 255, 0.32);
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4a65ff 0%, #6b95ff 100%);
        transform: translateY(-3px);
        box-shadow: 0 12px 38px rgba(94, 125, 255, 0.42);
    }

    .disclaimer {
        background: linear-gradient(135deg, rgba(94, 125, 255, 0.12) 0%, rgba(122, 163, 255, 0.10) 100%);
        border: 1.5px solid rgba(94, 125, 255, 0.28);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        font-size: 0.95rem;
        color: #2d3b7a;
        margin-top: 2.5rem;
        box-shadow: 0 10px 30px rgba(94, 125, 255, 0.12);
        line-height: 1.7;
    }

    .stat-box {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border: 1px solid #dee2e6;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }

    .stat-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .stat-box h3 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
        font-family: 'Roboto', sans-serif;
    }

    .stat-box p {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 8px 0 0 0;
        font-weight: 500;
    }

    .stFileUploader {
        background: linear-gradient(135deg, rgba(94, 125, 255, 0.08) 0%, rgba(122, 163, 255, 0.06) 100%);
        border: 2px dashed rgba(94, 125, 255, 0.5);
        border-radius: 18px;
        padding: 2.2rem;
        margin: 2rem 0;
        transition: all 0.35s ease;
    }

    .stFileUploader:hover {
        border-color: rgba(94, 125, 255, 0.8);
        background: linear-gradient(135deg, rgba(94, 125, 255, 0.14) 0%, rgba(122, 163, 255, 0.12) 100%);
        box-shadow: 0 8px 30px rgba(94, 125, 255, 0.18);
    }

    .sidebar-header {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
    }

    .sidebar-header h3 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
    }

    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #3498db, #e74c3c);
        margin: 1.5rem 0;
        border-radius: 1px;
    }

    .emoji-large {
        font-size: 2rem;
        margin-right: 0.5rem;
        display: inline-block;
    }

    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
        letter-spacing: -0.3px;
    }

    [data-testid="stMarkdownContainer"] h3, 
    [data-testid="stMarkdownContainer"] h4 {
        color: #1f2a59;
        font-weight: 700;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #4a5568;
        line-height: 1.7;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        .result-malignant, .result-benign, .result-normal {
            font-size: 1.5rem;
            padding: 1.5rem 2rem;
        }
        .stat-box h3 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CLASS CONFIGURATION
# ─────────────────────────────────────────────
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
# ⚠️ Update class order based on your model's training output
# Check train output: "Class mapping: {'Bengin cases': 0, 'Malignant cases': 1, 'Normal cases': 2}"

CLASS_INFO = {
    'Malignant': {
        'emoji': '🔴',
        'color': 'malignant',
        'description': 'Cancerous tumor detected. Malignant tumors are aggressive and can spread to other parts of the body. Immediate medical consultation is strongly advised.',
        'recommendation': 'Please consult an oncologist immediately for further diagnosis and treatment planning.'
    },
    'Benign': {
        'emoji': '🟡',
        'color': 'benign',
        'description': 'Non-cancerous growth detected. Benign tumors do not spread but should be monitored regularly by a medical professional.',
        'recommendation': 'Schedule a follow-up with your doctor for monitoring and further evaluation.'
    },
    'Normal': {
        'emoji': '🟢',
        'color': 'normal',
        'description': 'No cancerous abnormalities detected. The lung CT scan appears normal based on the analysis.',
        'recommendation': 'Continue regular health check-ups as recommended by your physician.'
    }
}

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model/lung_cancer_cnn.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ─────────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0          # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def is_grayscale_image(image: Image.Image, diff_threshold: int = 15, color_fraction: float = 0.05) -> bool:
    """Heuristic check: CT scan images are typically grayscale, not color photos."""
    rgb = np.array(image.convert("RGB"), dtype=np.int16)
    diff = np.abs(rgb[:, :, 0] - rgb[:, :, 1]) + np.abs(rgb[:, :, 1] - rgb[:, :, 2]) + np.abs(rgb[:, :, 0] - rgb[:, :, 2])
    color_pixels = np.count_nonzero(diff > diff_threshold)
    return (color_pixels / diff.size) < color_fraction


def validate_ct_scan_image(image: Image.Image) -> tuple[bool, str]:
    """Validate that the uploaded image resembles a CT scan image."""
    if not is_grayscale_image(image):
        return False, (
            "This image appears to be a color photograph rather than a lung CT scan. "
            "Please upload a valid grayscale CT scan image for reliable results."
        )
    return True, ""

# ─────────────────────────────────────────────
#  PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict(model, image: Image.Image):
    """Run prediction and return class name + confidence scores."""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    return predicted_class, confidence, predictions

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🫁 Lung Cancer Detection</h3></div>', unsafe_allow_html=True)
    
    st.markdown("""
    This system uses a **Convolutional Neural Network (CNN)** trained on the 
    **IQ-OTH/NCCD** lung cancer dataset to classify CT scan images into:
    
    - 🔴 **Malignant** — Cancerous
    - 🟡 **Benign** — Non-cancerous growth  
    - 🟢 **Normal** — No abnormality
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stat-box"><h3>561</h3><p>Malignant</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-box"><h3>416</h3><p>Normal</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box"><h3>120</h3><p>Benign</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-box"><h3>197</h3><p>Test Images</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    - **Base:** MobileNetV2 (Transfer Learning)
    - **Input:** 224×224 RGB images
    - **Output:** 3-class Softmax
    - **Trained with:** ImageNet weights
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Final Year Project** | CNN for Medical Imaging")

# ─────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="hero-accent"></div>
    <div class="hero-accent-small"></div>
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-tag">Lung CT Scan Analysis</div>
            <h1>Advanced Lung Cancer Detection</h1>
            <p class="hero-copy">Upload a CT scan and let our deep-learning system identify suspicious lung areas, provide clear risk guidance, and support a professional review workflow.</p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <strong>3-class prediction</strong>
                    <span>Malignant, Benign, or Normal</span>
                </div>
                <div class="hero-stat">
                    <strong>CT scan ready</strong>
                    <span>Optimized for grayscale medical imaging</span>
                </div>
                <div class="hero-stat">
                    <strong>Fast insights</strong>
                    <span>Results in seconds with intuitive feedback</span>
                </div>
            </div>
        </div>
        <div class="hero-illustration">
            <svg viewBox="0 0 520 520" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Medical lung scan illustration">
                <defs>
                    <linearGradient id="panelGrad" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#1f2a59" />
                        <stop offset="100%" stop-color="#4c7dff" />
                    </linearGradient>
                    <linearGradient id="lungFill" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#9fc6ff" />
                        <stop offset="100%" stop-color="#5b84ff" />
                    </linearGradient>
                    <linearGradient id="lineGlow" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="rgba(255,255,255,0)" />
                        <stop offset="50%" stop-color="rgba(255,255,255,0.35)" />
                        <stop offset="100%" stop-color="rgba(255,255,255,0)" />
                    </linearGradient>
                    <radialGradient id="alertGlow" cx="50%" cy="50%" r="60%">
                        <stop offset="0%" stop-color="#ff718d" stop-opacity="0.9" />
                        <stop offset="100%" stop-color="#ff718d" stop-opacity="0" />
                    </radialGradient>
                </defs>
                <rect x="20" y="20" width="480" height="480" rx="48" fill="url(#panelGrad)" />
                <rect x="56" y="60" width="408" height="332" rx="28" fill="#15213f" opacity="0.94" />
                <g opacity="0.24" stroke="#ffffff" stroke-width="1">
                    <path d="M76 96 H456" />
                    <path d="M76 136 H456" />
                    <path d="M76 176 H456" />
                    <path d="M76 216 H456" />
                    <path d="M76 256 H456" />
                    <path d="M76 296 H456" />
                    <path d="M76 336 H456" />
                    <path d="M76 376 H456" />
                </g>
                <g opacity="0.16" stroke="#ffffff" stroke-width="1">
                    <path d="M96 80 V420" />
                    <path d="M136 80 V420" />
                    <path d="M176 80 V420" />
                    <path d="M216 80 V420" />
                    <path d="M256 80 V420" />
                    <path d="M296 80 V420" />
                    <path d="M336 80 V420" />
                    <path d="M376 80 V420" />
                </g>
                <path d="M170 118 C 128 142 114 224 154 300 C 162 320 186 326 212 320 C 238 314 248 236 228 172 C 218 142 198 118 170 118 Z"
                      fill="url(#lungFill)" opacity="0.96" />
                <path d="M350 126 C 388 154 402 234 366 304 C 358 320 334 328 306 324 C 282 320 276 244 292 178 C 304 142 324 122 350 126 Z"
                      fill="url(#lungFill)" opacity="0.96" />
                <path d="M182 142 C 206 176 214 216 206 254" stroke="rgba(255,255,255,0.65)" stroke-width="10" fill="none" stroke-linecap="round" />
                <path d="M330 152 C 356 188 364 228 352 266" stroke="rgba(255,255,255,0.65)" stroke-width="10" fill="none" stroke-linecap="round" />
                <path d="M100 420 H420" stroke="rgba(255,255,255,0.12)" stroke-width="18" stroke-linecap="round" />
                <rect x="72" y="406" width="70" height="20" rx="10" fill="rgba(255,255,255,0.08)" />
                <rect x="132" y="384" width="220" height="12" rx="6" fill="rgba(255,255,255,0.1)" />
                <circle cx="248" cy="210" r="30" fill="url(#alertGlow)" />
                <circle cx="256" cy="210" r="12" fill="#ff718d" />
                <circle cx="360" cy="250" r="16" fill="#7ad0ff" opacity="0.9" />
                <circle cx="180" cy="260" r="12" fill="#7ad0ff" opacity="0.95" />
            </svg>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error("⚠️ Model file not found! Please train the model first by running `python train_model.py`")
    st.info("After training, the model will be saved as `model/lung_cancer_cnn.h5`")
    st.stop()

st.success("✅ Model loaded successfully!")

# ─────────────────────────────────────────────
#  UPLOAD & PREDICT
# ─────────────────────────────────────────────
st.markdown("### 📤 Upload CT Scan Image")

uploaded_file = st.file_uploader(
    "Choose a CT Scan image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a lung CT scan image for analysis. Color photographs are not valid inputs."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    valid_image, validation_message = validate_ct_scan_image(image)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🖼️ Uploaded CT Scan")
        st.image(image, use_column_width=True, caption="Input CT Scan Image")

    with col2:
        st.markdown("#### 🔬 Analysis Results")

        if not valid_image:
            st.error("This image is not a valid CT scan image. Please upload a proper lung CT scan in grayscale format.")
            st.warning(validation_message)
        else:
            with st.spinner("Analyzing CT scan... Please wait"):
                time.sleep(1)
                predicted_class, confidence, all_probs = predict(model, image)
            info = CLASS_INFO[predicted_class]

            # Main result badge
            st.markdown(
                f'<div class="result-{info["color"]}">'
                f'<span class="emoji-large">{info["emoji"]}</span> {predicted_class.upper()}'
                f'<br><span class="confidence-text">Confidence: {confidence:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # All class probabilities
            st.markdown("**Confidence Scores for All Classes:**")
            for i, cls in enumerate(CLASS_NAMES):
                prob = all_probs[i] * 100
                bar_color = "#ff4757" if cls == "Malignant" else "#f39c12" if cls == "Benign" else "#2ecc71"
                st.markdown(f"**{cls}:** {prob:.1f}%")
                st.markdown(
                    f'<div class="confidence-bar-container">'
                    f'<div style="width:{prob}%;height:100%;background:{bar_color};border-radius:999px;transition:width 0.5s;"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Info and recommendation
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f"**ℹ️ What this means:**\n\n{info['description']}")
            st.markdown(f"**💊 Recommendation:**\n\n{info['recommendation']}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is developed for educational and research purposes 
        as part of a final year academic project. It is NOT a substitute for professional medical diagnosis. 
        Always consult a qualified radiologist or oncologist for medical decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # Demo placeholder
    st.info("👆 Please upload a CT scan image to begin the analysis.")

    st.markdown("### 💡 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1: Upload**
        Upload a lung CT scan image in JPG or PNG format using the uploader above.
        """)
    with col2:
        st.markdown("""
        **Step 2: Analyze**
        The CNN model processes the image through multiple convolutional layers to extract features.
        """)
    with col3:
        st.markdown("""
        **Step 3: Result**
        The model classifies the scan as Malignant, Benign, or Normal with a confidence score.
        """)
