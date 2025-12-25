import streamlit as st

# WAJIB PALING ATAS
st.set_page_config(
    page_title="3D CNN Model",
    layout="wide"
)

# BARU SETELAH ITU BOLEH ADA KODE LAIN
st.markdown("""
<style>
.block-container .stColumn {
    padding-left: 0rem;
    padding-right: 0rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 3D CNN, Dilated Convolution, dan CBAM untuk Klasifikasi MRI ADHD")

st.markdown("""
Model ini menggunakan pendekatan **3D Convolutional Neural Network (3D CNN)** untuk memproses citra MRI 3D secara utuh.
""")


# --- 3D CNN ---
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("1️⃣ 3D Convolutional Neural Network")
    st.markdown("""
3D CNN memungkinkan model menangkap hubungan spasial tiga dimensi di citra MRI. Dengan memproses keseluruhan volume otak, model dapat mengekstrak pola struktural kompleks yang mungkin terkait dengan ADHD. Hal ini meningkatkan kemampuan model dalam mendeteksi fitur volumetrik tersebar.
""")
    with st.expander("💡 Detail tambahan 3D CNN"):
        st.markdown("""
Blok Conv3D belajar filter 3D untuk mendeteksi fitur lokal. BatchNormalization membantu stabilisasi training, ReLU memperkenalkan non-linearitas, dan MaxPooling3D mengurangi dimensi spasial sambil mempertahankan fitur penting.
""")
with col2:
    st.image("images/3d_cnn.png", caption="Ilustrasi 3D CNN memproses volume MRI", width=500)

# --- Dilated Convolution ---
col1, col2 = st.columns([1,1])
with col1:
    st.image("images/dilated_conv.png", caption="Skema Dilated Convolution", width=500)
with col2:
    st.subheader("2️⃣ Dilated Convolution")
    st.markdown("""
Dilated Convolution memperluas **receptive field** tanpa menambah jumlah parameter, memungkinkan model mempelajari konteks global di citra MRI. Teknik ini sangat berguna untuk mendeteksi pola ADHD yang tersebar luas, sehingga fitur penting tetap terjaga meski resolusi spasial dipertahankan.
""")
    with st.expander("💡 Detail tambahan Dilated Convolution"):
        st.markdown("""
Dengan dilasi, filter ‘melompat’ melewati beberapa voxel, menangkap informasi lebih luas tanpa pooling tambahan. Ini membantu model mengenali hubungan jarak jauh dalam volume otak.
""")

# --- CBAM ---
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("3️⃣ CBAM (Convolutional Block Attention Module)")
    st.markdown("""
CBAM adalah modul attention yang memfokuskan model pada fitur paling relevan. Terdiri dari **channel attention** yang menekankan saluran fitur penting dan **spatial attention** yang menyoroti lokasi kritis di citra 3D. Dengan CBAM, model dapat meningkatkan performa klasifikasi.
""")
    with st.expander("💡 Detail tambahan CBAM"):
        st.markdown("""
Channel Attention menghitung importance tiap channel dengan global pooling, sementara Spatial Attention menghitung importance tiap voxel dalam fitur map. Kombinasi ini membantu model fokus pada informasi penting dan mengabaikan noise.
""")
with col2:
    st.image("images/cbam.png", caption="Ilustrasi CBAM dengan Channel & Spatial Attention", width=500)

st.markdown("""
✨ Dengan kombinasi **3D CNN**, **Dilated Convolution**, dan **CBAM**, model mampu mengekstraksi fitur volumetrik otak yang relevan, menangkap konteks global, serta fokus pada saluran dan lokasi penting. Hal ini membuat klasifikasi MRI ADHD lebih akurat dan interaktif bagi pengguna.
""")

