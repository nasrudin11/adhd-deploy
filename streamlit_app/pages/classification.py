import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling3D, GlobalMaxPooling3D, Dense,
    Multiply, Add, Reshape, Lambda, Concatenate, Conv3D
)
from tensorflow.keras.models import load_model
from skimage.transform import resize
import os

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model_3d():
    return load_model("models/model_3dcnn.h5", compile=False)

model = load_model_3d()

# =========================
# Preprocess MRI
# =========================
def preprocess_mri(path, target_shape=(128, 128, 128)):
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

    vol_input = np.expand_dims(vol_norm, axis=-1)
    vol_input = np.expand_dims(vol_input, axis=0)

    return vol_input, vol_norm

# =========================
# UI
# =========================
st.title("🧠 3D MRI ADHD Classification")

uploaded = st.file_uploader(
    "Upload MRI (.nii / .nii.gz)",
    type=["nii", "nii.gz"]
)

# =========================
# Prediction
# =========================
prob_tdc = float(model.predict(X)[0][0])
prob_adhd = 1.0 - prob_tdc

# =========================
# Probability Section
# =========================
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

# =========================
# Classification Result
# =========================
st.subheader("🔍 Hasil Klasifikasi")

if prob_tdc >= 0.5:
    st.success(
        f"🟢 **TDC (Typhical Developing Children)** terdeteksi\n\n"
        f"Probabilitas TDC = **{prob_tdc:.4f}**"
    )
else:
    st.warning(
        f"🟠 **ADHD (Attention Deficit Hyperactive Disorder)** terdeteksi\n\n"
        f"Probabilitas ADHD = **{prob_adhd:.4f}**"
    )

finally:
if os.path.exists(temp_path):
    os.remove(temp_path)

