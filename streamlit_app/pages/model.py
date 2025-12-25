import streamlit as st
import os

# ============================================================
# KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# ============================================================
st.set_page_config(
    page_title="3D CNN Model",
    layout="wide"
)

# ============================================================
# PATH SETUP (AMAN UNTUK STREAMLIT CLOUD)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "..", "images")

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.title("🧠 3D CNN, Dilated Convolution, dan CBAM untuk Klasifikasi MRI ADHD")

st.markdown("""
Model ini menggunakan pendekatan **3D Convolutional Neural Network (3D CNN)** untuk memproses citra MRI 3D secara utuh.
Pendekatan ini memungkinkan ekstraksi fitur spasial volumetrik yang penting dalam analisis gangguan neurodevelopmental seperti ADHD.
""")

# ============================================================
# 3D CNN SECTION
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣ 3D Convolutional Neural Network")
    st.markdown("""
3D CNN memungkinkan model menangkap hubungan spasial tiga dimensi pada citra MRI.
Dengan memproses volume penuh, model dapat mengekstraksi pola struktural kompleks
yang tidak dapat ditangkap oleh CNN 2D biasa.
""")

    with st.expander("💡 Detail tambahan 3D CNN"):
        st.markdown("""
- Menggunakan kernel 3D untuk menangkap informasi spasial volumetrik  
- Batch Normalization menjaga stabilitas training  
- ReLU meningkatkan non-linearitas  
- MaxPooling3D mengurangi dimensi spasial  
""")

with col2:
    st.image(
        os.path.join(IMAGE_DIR, "3d_cnn.png"),
        caption="Ilustrasi 3D CNN memproses volume MRI",
        width=500
    )

# ============================================================
# DILATED CONVOLUTION
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.image(
        os.path.join(IMAGE_DIR, "dilated_conv.png"),
        caption="Skema Dilated Convolution",
        width=500
    )

with col2:
    st.subheader("2️⃣ Dilated Convolution")
    st.markdown("""
Dilated Convolution memperluas **receptive field** tanpa menambah jumlah parameter.
Teknik ini sangat efektif untuk menangkap konteks global pada citra MRI tanpa
kehilangan resolusi spasial.
""")

    with st.expander("💡 Detail tambahan Dilated Convolution"):
        st.markdown("""
Dengan memperbesar jarak antar kernel, dilated convolution memungkinkan model
mengamati area yang lebih luas tanpa pooling berlebih.
""")

# ============================================================
# CBAM
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("3️⃣ CBAM (Convolutional Block Attention Module)")
    st.markdown("""
CBAM meningkatkan performa model dengan memfokuskan perhatian pada fitur yang
paling relevan melalui dua mekanisme:
- **Channel Attention**
- **Spatial Attention**
""")

    with st.expander("💡 Detail tambahan CBAM"):
        st.markdown("""
Channel Attention menekankan fitur penting antar channel,
sementara Spatial Attention menyoroti lokasi penting dalam volume citra.
""")

with col2:
    st.image(
        os.path.join(IMAGE_DIR, "cbam.png"),
        caption="Ilustrasi CBAM (Channel & Spatial Attention)",
        width=500
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
---
✨ Dengan kombinasi **3D CNN**, **Dilated Convolution**, dan **CBAM**, model mampu
mengekstraksi fitur volumetrik yang kaya, memahami konteks global, serta meningkatkan
akurasi klasifikasi MRI ADHD secara signifikan.
""")
