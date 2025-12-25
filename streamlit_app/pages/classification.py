# =========================
# IMPORTS
# =========================
import streamlit as st
import numpy as np
import nibabel as nib
from skimage.transform import resize
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Conv3D, Dense, GlobalAveragePooling3D, GlobalMaxPooling3D,
    Reshape, Add, Multiply, Concatenate, Activation, BatchNormalization, ReLU
)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_3dcnn.keras")

@st.cache_resource
def load_model_3d():
    return load_model(MODEL_PATH, compile=False)

model = load_model_3d()

# =========================
# PREPROCESS MRI
# =========================
def preprocess_mri(path, target_shape=(128,128,128)):
    vol = nib.load(path).get_fdata()

    if vol.shape != target_shape:
        vol = resize(
            vol,
            target_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True
        )

    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol_norm = vol_norm.astype(np.float32)

    vol_input = np.expand_dims(vol_norm, axis=-1)  # (128,128,128,1)
    vol_input = np.expand_dims(vol_input, axis=0)  # (1,128,128,128,1)

    return vol_input, vol_norm

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="3D MRI ADHD Classification", layout="wide")
st.title("🧠 3D MRI ADHD Classification")

uploaded = st.file_uploader(
    "Upload MRI (.nii / .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded is not None:
    temp_path = f"temp_{uploaded.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        vol_input, vol_norm = preprocess_mri(temp_path)

        # Prediction
        with st.spinner("Memproses MRI..."):
            prob_tdc = float(model.predict(vol_input)[0][0])
            prob_adhd = 1.0 - prob_tdc

        # Probability section
        st.subheader("📊 Probabilitas Kelas (Sigmoid Output)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TDC (Typical Development Control)**")
            st.progress(prob_tdc)
            st.caption(f"{prob_tdc:.4f}")
        with col2:
            st.markdown("**ADHD**")
            st.progress(prob_adhd)
            st.caption(f"{prob_adhd:.4f}")

        # Classification result
        st.subheader("🔍 Hasil Klasifikasi")
        if prob_tdc >= 0.5:
            st.success(f"🟢 **TDC (Typical Developing Children)** terdeteksi\nProbabilitas TDC = **{prob_tdc:.4f}**")
        else:
            st.warning(f"🟠 **ADHD (Attention Deficit Hyperactive Disorder)** terdeteksi\nProbabilitas ADHD = **{prob_adhd:.4f}**")

    except Exception as e:
        st.error(f"Error saat memproses MRI: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)





