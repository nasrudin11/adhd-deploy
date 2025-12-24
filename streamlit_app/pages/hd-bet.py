import streamlit as st
import nibabel as nib
import numpy as np
from skimage.transform import resize
from hd_bet import run_hd_bet
import os

st.title("🧠 Coba HD-BET dengan Streamlit")

uploaded = st.file_uploader("Upload MRI (.nii / .nii.gz)", type=["nii", "nii.gz"])

if uploaded is not None:
    temp_path = f"temp_{uploaded.name}"
    output_path = f"bet_{uploaded.name}"

    # Simpan file sementara
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Menjalankan HD-BET untuk skull stripping...")

    # Jalankan HD-BET (CPU mode)
    with st.spinner("Processing..."):
        run_hd_bet(input_file=temp_path, output_file=output_path, device='cpu')

    st.success("Skull stripping selesai!")

    # Tampilkan beberapa slice tengah
    vol = nib.load(output_path).get_fdata()
    mid_slices = [vol.shape[i]//2 for i in range(3)]  # slice tengah sumbu x,y,z

    st.subheader("Slice tengah (axial, coronal, sagittal)")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(vol[mid_slices[0], :, :], cmap='gray')
    axes[0].set_title('Sagittal')
    axes[1].imshow(vol[:, mid_slices[1], :], cmap='gray')
    axes[1].set_title('Coronal')
    axes[2].imshow(vol[:, :, mid_slices[2]], cmap='gray')
    axes[2].set_title('Axial')

    for ax in axes:
        ax.axis('off')

    st.pyplot(fig)

    # Hapus file sementara
    os.remove(temp_path)
    os.remove(output_path)
