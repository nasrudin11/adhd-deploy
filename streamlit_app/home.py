import streamlit as st

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Tentang ADHD",
    layout="wide"
)

st.title("📘 Tentang ADHD")

# =========================
# Deskripsi ADHD
# =========================
st.markdown("""
**Attention Deficit Hyperactivity Disorder (ADHD)** merupakan gangguan neurodevelopmental
yang ditandai oleh gangguan perhatian (*inattention*), kontrol impuls (*impulsivity*),
dan regulasi aktivitas motorik (*hyperactivity*).

Gangguan ini umumnya mulai terdeteksi pada masa kanak-kanak dan pada sebagian individu
dapat berlanjut hingga usia dewasa.  
Secara neurobiologis, ADHD berkaitan dengan perbedaan struktur dan fungsi pada beberapa area otak,
terutama **prefrontal cortex**, **basal ganglia**, dan **cerebellum**,
yang berperan penting dalam fungsi eksekutif, pengendalian perilaku, dan pengambilan keputusan.
""")

st.caption(
    "Informasi ini disajikan sebagai latar belakang klinis untuk sistem klasifikasi ADHD berbasis citra MRI."
)

# =========================
# Gejala Utama
# =========================
st.subheader("🧩 Gejala Utama")

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        "🧠 **Inattention**\n"
        "- Sulit mempertahankan fokus\n"
        "- Mudah terdistraksi\n"
        "- Sering lupa tugas atau aktivitas"
    )

with col2:
    st.warning(
        "⚡ **Hyperactivity**\n"
        "- Gelisah atau tidak bisa diam\n"
        "- Sulit duduk dalam waktu lama\n"
        "- Aktivitas motorik berlebihan"
    )

with col3:
    st.info(
        "💥 **Impulsivity**\n"
        "- Menyela pembicaraan\n"
        "- Kesulitan menunggu giliran\n"
        "- Mengambil keputusan secara terburu-buru"
    )

# =========================
# Area Otak Terkait
# =========================
st.subheader("🧠 Area Otak Terkait")

st.markdown("""
- **Prefrontal Cortex**  
  Berperan dalam perhatian, perencanaan, dan pengambilan keputusan.

- **Basal Ganglia**  
  Terlibat dalam kontrol motorik, regulasi perilaku, dan sistem reward.

- **Cerebellum**  
  Berperan dalam koordinasi motorik serta fungsi kognitif dan emosional.
""")

# =========================
# Faktor Penyebab
# =========================
st.subheader("⚠️ Faktor Penyebab")

st.markdown("""
- **Faktor genetik**  
  ADHD memiliki kecenderungan diturunkan dalam keluarga.

- **Faktor lingkungan**  
  Paparan toksin, alkohol, atau rokok selama kehamilan.

- **Perkembangan otak**  
  Keterlambatan maturasi pada area otak tertentu yang berperan dalam kontrol perhatian dan perilaku.
""")

# =========================
# Prevalensi
# =========================
st.subheader("🌍 Prevalensi")

st.markdown("""
- Sekitar **5–7%** anak-anak dan **2–5%** orang dewasa di seluruh dunia mengalami ADHD.  
- Lebih sering terdiagnosis pada anak laki-laki dibandingkan perempuan.
""")

# =========================
# Komorbiditas
# =========================
st.subheader("🔗 Komorbiditas")

st.markdown("""
ADHD sering muncul bersamaan dengan gangguan lain, antara lain:
- Gangguan kecemasan dan depresi  
- Gangguan belajar  
- Tourette syndrome
""")

# =========================
# Penanganan
# =========================
st.subheader("💊 Penanganan")

st.markdown("""
Pendekatan penanganan ADHD umumnya bersifat multidisipliner, meliputi:
- Terapi perilaku  
- Intervensi pendidikan  
- Penggunaan obat-obatan stimulant maupun non-stimulant  
- Dukungan keluarga dan lingkungan
""")
