# =========================
# IMPORTS
# =========================
import streamlit as st
import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import measure
import plotly.graph_objects as go
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# =========================
# CUSTOM LAYER
# =========================
@register_keras_serializable()
class ChannelPool3D(tf.keras.layers.Layer):
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        return tf.concat([avg_pool, max_pool], axis=-1)1)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_3dcnn.keras")

@st.cache_resource
def load_model_3d():
    # Custom layer sudah diregister, aman load tanpa custom_objects
    return load_model(
            MODEL_PATH,
            custom_objects={
                "ChannelPool3D": ChannelPool3D
            }

    )

model = load_model_3d()


# =========================
# CROPPING & PADDING FUNCTIONS
# =========================
def crop_to_brain_bbox(vol, margin=5):
    coords = np.where(vol > 0)
    if len(coords[0]) == 0:
        return vol
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    z_min, y_min, x_min = max(0, z_min - margin), max(0, y_min - margin), max(0, x_min - margin)
    z_max, y_max, x_max = min(vol.shape[0], z_max + margin), min(vol.shape[1], y_max + margin), min(vol.shape[2], x_max + margin)
    return vol[z_min:z_max, y_min:y_max, x_min:x_max]

def crop_or_pad_depth(vol, target=128):
    D = vol.shape[0]
    if D > target:
        cut_total = D - target
        cut_before = cut_total // 2
        cut_after = cut_total - cut_before
        vol = vol[cut_before:D - cut_after]
    elif D < target:
        pad_total = target - D
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        vol = np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
    return vol

# =========================
# FULL PREPROCESS FUNCTION
# =========================
def preprocess_mri(path, target_shape=(128,128,128)):
    vol_raw = nib.load(path).get_fdata().astype(np.float32)  # raw MRI
    vol = vol_raw.copy()  # untuk preprocessing

    # 1️⃣ Crop ke bounding box otak
    vol = crop_to_brain_bbox(vol, margin=5)

    # 2️⃣ Resize HxW
    vol = np.stack([resize(vol[i], target_shape[:2], order=1, preserve_range=True, anti_aliasing=True)
                    for i in range(vol.shape[0])]).astype(np.float32)

    # 3️⃣ Crop / pad depth
    vol = crop_or_pad_depth(vol, target=target_shape[0])

    # 4️⃣ Normalisasi
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

    # 5️⃣ Untuk model input
    vol_model = np.expand_dims(np.moveaxis(vol, 0, 2), axis=-1)  # (H,W,D,1)
    vol_model = np.expand_dims(vol_model, axis=0)  # (1,H,W,D,1)

    return vol_model, vol, vol_raw

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="3D MRI ADHD Classification", layout="centered")
st.title("🧠 3D MRI ADHD Classification")

uploaded = st.file_uploader("Upload MRI (.nii / .nii.gz)", type=["nii", "nii.gz"])

if uploaded is not None:
    temp_path = f"temp_{uploaded.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        vol_input, vol_pre, vol_raw = preprocess_mri(temp_path)

        # =========================
        # VISUALISASI 3D - MARCHING CUBES
        # =========================

        st.subheader("🧊 Visualisasi 3D MRI")
        st.markdown("""
        Menampilkan bentuk **3D otak** hasil rekonstruksi menggunakan metode  
        **Marching Cubes** untuk memberikan gambaran struktur anatomi secara menyeluruh.
        """)

        # Gunakan volume asli (tanpa downsampling)
        vol_mc = vol_raw

        # Threshold otomatis untuk menghindari background
        threshold = np.percentile(vol_mc, 60)

        # Marching Cubes
        verts, faces, _, _ = measure.marching_cubes(
            vol_mc,
            level=threshold
        )

        # Membuat mesh 3D
        mesh = go.Mesh3d(
            # Koordinat x, y, z setiap vertex dari mesh (titik 3D)
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            
            # Indeks vertex tiap segitiga/face
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            
            # Warna mesh
            color='gray',
            
            # Transparansi
            opacity=1,
            
            # Pengaturan pencahayaan permukaan mesh
            lighting=dict(
                ambient=0.3,   # cahaya global/sekitar
                diffuse=0.7,   # cahaya dari sumber utama (shading)
                specular=0.4,  # pantulan cahaya (kilau)
                roughness=0.6  # kekasaran permukaan (matte vs glossy)
            )
        )



        # Plot
        # Membuat figure 3D dari mesh yang sudah dibuat
        fig3d = go.Figure(mesh)
        # Mengatur layout figure
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            # Mengatur margin figure (kiri, kanan, atas, bawah)
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # use_container_width=True → otomatis menyesuaikan lebar figure dengan container Streamlit
        st.plotly_chart(fig3d, use_container_width=True)


        # =========================
        # VISUALISASI RAW MRI
        # =========================
        st.subheader("🧩 Visualisasi 2D MRI")
        st.markdown("""
        Visualisasi irisan **Axial, Coronal, dan Sagittal** dari citra MRI asli
        sebelum dilakukan proses resize.
        """)
        X, Y, Z = vol_raw.shape
        axial   = vol_raw[:, :, Z//2]
        coronal = vol_raw[:, Y//2, :]
        sagittal= vol_raw[X//2, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(axial.T, cmap='gray', origin='lower'); axes[0].set_title("Axial"); axes[0].axis('off')
        axes[1].imshow(coronal.T, cmap='gray', origin='lower'); axes[1].set_title("Coronal"); axes[1].axis('off')
        axes[2].imshow(sagittal.T, cmap='gray', origin='lower'); axes[2].set_title("Sagittal"); axes[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # =========================
        # VISUALISASI AFTER RESIZE
        # =========================
        st.subheader("🧩 Visualisasi Resize")
        st.markdown(""" Visualisasi hasil setelah dilakukan proses resize image citra MRI """)
        X, Y, Z = vol_pre.shape
        axial   = vol_pre[:, :, Z//2]
        coronal = vol_pre[:, Y//2, :]
        sagittal= vol_pre[X//2, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(axial.T, cmap='gray', origin='lower'); axes[0].set_title("Axial"); axes[0].axis('off')
        axes[1].imshow(coronal.T, cmap='gray', origin='lower'); axes[1].set_title("Coronal"); axes[1].axis('off')
        axes[2].imshow(sagittal.T, cmap='gray', origin='lower'); axes[2].set_title("Sagittal"); axes[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # =========================
        # PREDIKSI
        # =========================
        with st.spinner("Memproses MRI..."):
            prob_tdc = float(model.predict(vol_input)[0][0])
            prob_adhd = 1.0 - prob_tdc

        st.subheader("📊 Probabilitas Kelas (Sigmoid Output)")
        st.markdown("""
        Nilai probabilitas menunjukkan **tingkat keyakinan model** terhadap masing-masing kelas.
        Semakin besar nilainya, semakin tinggi keyakinan model.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TDC (Typical Development Control)**")
            st.progress(prob_tdc)
            st.caption(f"{prob_tdc:.4f} → Model yakin {prob_tdc*100:.1f}% bahwa ini TDC")
        with col2:
            st.markdown("**ADHD**")
            st.progress(prob_adhd)
            st.caption(f"{prob_adhd:.4f} → Model yakin {prob_adhd*100:.1f}% bahwa ini ADHD")

        st.subheader("🔍 Hasil Klasifikasi")
        if prob_tdc >= prob_adhd:
            st.success(
                f"🟢 **Prediksi: TDC (Typical Developing Children)**\n"
                f"Model memprediksi bahwa MRI ini lebih mirip TDC karena probabilitas TDC ({prob_tdc*100:.1f}%) "
                f"lebih tinggi dibandingkan probabilitas ADHD ({prob_adhd*100:.1f}%)."
            )
        else:
            st.warning(
                f"🟠 **Prediksi: ADHD (Attention Deficit Hyperactive Disorder)**\n"
                f"Model memprediksi bahwa MRI ini lebih mirip ADHD karena probabilitas ADHD ({prob_adhd*100:.1f}%) "
                f"lebih tinggi dibandingkan probabilitas TDC ({prob_tdc*100:.1f}%)."
            )


    except Exception as e:
        st.error(f"Error saat memproses MRI: {e}")

    finally:
        if os.path.exists(temp_path): 
            os.remove(temp_path) 


















