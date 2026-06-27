import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import io

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="G2G Shade Finder",
    page_icon="💄",
    layout="wide",
)

# ─────────────────────────────────────────
# CUSTOM CSS – TEMA PINK FEMININ
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Root variables */
:root {
    --pink-blush:   #F9C6D0;
    --pink-hot:     #E8638C;
    --pink-soft:    #FDE8EF;
    --pink-deep:    #C2185B;
    --cream:        #FFF8FA;
    --text-dark:    #3D1A25;
    --text-mid:     #7D4455;
    --radius:       16px;
}

/* Background */
.stApp { background-color: var(--cream); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #F9C6D0 0%, #FADADD 60%, #FDE8EF 100%);
    border-right: 1px solid #F0A0B8;
}
[data-testid="stSidebar"] * { color: var(--text-dark) !important; }

/* Typography */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--pink-deep) !important;
}
p, li, label, div {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(232,99,140,0.08);
}
[data-testid="stMetricValue"] {
    color: var(--pink-deep) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #E8638C, #C2185B);
    color: white !important;
    border: none;
    border-radius: 50px;
    padding: 10px 28px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Uploader */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed var(--pink-blush);
    border-radius: var(--radius);
    padding: 16px;
}

/* Info / success / warning boxes */
.stAlert { border-radius: var(--radius) !important; }

/* Divider */
hr { border-color: var(--pink-blush); }

/* Badge helper classes */
.badge-ok {
    display:inline-block; background:#E8F5E9; color:#2E7D32;
    border-radius:50px; padding:3px 12px; font-size:0.82rem; font-weight:500;
}
.badge-warn {
    display:inline-block; background:#FFF3E0; color:#E65100;
    border-radius:50px; padding:3px 12px; font-size:0.82rem; font-weight:500;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #FDE8EF, #fff);
    border: 1.5px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-top: 16px;
    box-shadow: 0 4px 20px rgba(232,99,140,0.12);
}
.result-card h2 { margin-bottom: 4px; }

/* Recommendation card */
.rec-card {
    background: white;
    border-left: 4px solid var(--pink-hot);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Camera container */
.camera-container {
    background: white;
    border: 2px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 20px;
    margin-top: 16px;
}

.camera-container .stButton {
    text-align: center;
}

/* Validation status */
.validation-pass {
    background: #E8F5E9;
    border-left: 4px solid #2E7D32;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}

.validation-fail {
    background: #FFF3E0;
    border-left: 4px solid #E65100;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────
CLASS_NAMES = ["dark", "fair", "light"]          # sesuaikan urutan folder dataset
IMG_SIZE    = (224, 224)
MODEL_PATH  = "skin_tone_model.h5"

# G2G shade mapping per kelas skin tone
G2G_RECOMMENDATION = {
    "fair": {
        "shades":   "00 Allegato / 01 Buttercream",
        "skincare": "SPF 50+, Brightening Serum, Vitamin C",
        "tip":      "Kulit fair rentan sunburn – pastikan pakai sunscreen setiap hari!",
    },
    "light": {
        "shades":   "01 Buttercream / 02 Praline",
        "skincare": "SPF 50+, Niacinamide, Hydrating Toner",
        "tip":      "Tone kulit light cocok dengan nuansa nude-pink untuk tampilan natural.",
    },
    "dark": {
        "shades":   "04 Ginger / 05 Cinnamon",
        "skincare": "Moisturizing Cream, SPF 30+, Shea Butter",
        "tip":      "Kulit dark terlihat glowing dengan foundation yang punya undertone warm.",
    },
}

# ─────────────────────────────────────────
# HELPER: Load model (cache agar tidak reload tiap interaksi)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model .h5 dari repo."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────
# HELPER: Preprocessing gambar
# ─────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize, normalize, expand dims → siap masuk model."""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalisasi [0,1]
    return np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)

# ─────────────────────────────────────────
# HELPER: Cek kualitas gambar
# ─────────────────────────────────────────
def check_image_quality(pil_img: Image.Image):
    """Return dict dengan status pencahayaan & deteksi wajah."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    # Cek pencahayaan
    if brightness < 80:
        light_status, light_msg = "warn", f"Terlalu gelap (brightness {brightness:.0f})"
    elif brightness > 200:
        light_status, light_msg = "warn", f"Terlalu terang (brightness {brightness:.0f})"
    else:
        light_status, light_msg = "ok", f"Pencahayaan baik (brightness {brightness:.0f})"

    # Deteksi wajah dengan Haar Cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        face_status, face_msg = "ok",   "Wajah terdeteksi"
    else:
        face_status, face_msg = "warn", "Wajah tidak terdeteksi – coba foto lebih dekat"

    return {
        "light_status": light_status, "light_msg": light_msg,
        "face_status":  face_status,  "face_msg":  face_msg,
    }

# ─────────────────────────────────────────
# HELPER: Validasi gambar untuk proses
# ─────────────────────────────────────────
def validate_image_for_processing(pil_img: Image.Image):
    """Validasi apakah gambar layak diproses."""
    quality = check_image_quality(pil_img)
    is_valid = (quality["light_status"] == "ok" and quality["face_status"] == "ok")
    return is_valid, quality

# ─────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## G2G Shade Finder")
    st.markdown("*AI-powered skin tone detection*")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["Upload Image", "Model Insight", "Beauty Recommendation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Final Project · Machine Learning · 2025")

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction    = None   # label string
    st.session_state.probabilities = None   # array float
    st.session_state.uploaded_img  = None   # PIL Image
    st.session_state.camera_image  = None   # PIL Image from camera
    st.session_state.auto_process  = False  # Flag untuk auto-process setelah validasi

model = load_model()

# ═══════════════════════════════════════════════════════════════
# HALAMAN 1 – UPLOAD IMAGE
# ═══════════════════════════════════════════════════════════════
if page == "Upload Image":
    st.markdown("# Temukan Shade-mu")
    st.markdown("Upload foto wajah atau ambil foto langsung menggunakan kamera.")

    if model is None:
        st.error("Model belum ditemukan. Pastikan `skin_tone_model.h5` ada di repo.")
        st.stop()

    # ── Tab untuk upload dan camera ──
    tab1, tab2 = st.tabs(["Upload Foto", "Ambil Foto"])

    # ── TAB 1: Upload ──
    with tab1:
        uploaded = st.file_uploader("Upload foto wajah (JPG / PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded:
            pil_img = Image.open(uploaded)
            st.session_state.uploaded_img = pil_img
            st.session_state.camera_image = None
            
            col_img, col_info = st.columns([1, 1], gap="large")

            with col_img:
                st.markdown("#### Preview")
                st.image(pil_img, use_container_width=True)

            with col_info:
                # Cek kualitas gambar
                st.markdown("#### Cek Kualitas Foto")
                quality = check_image_quality(pil_img)
                
                is_valid = (quality["light_status"] == "ok" and quality["face_status"] == "ok")
                
                if is_valid:
                    st.markdown(f"""
                    <div class="validation-pass">
                        <strong>✓</strong> {quality["light_msg"]}<br>
                        <strong>✓</strong> {quality["face_msg"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="validation-fail">
                        <strong>⚠</strong> {quality["light_msg"] if quality["light_status"] == "warn" else quality["face_msg"] if quality["face_status"] == "warn" else "Perbaiki kualitas foto"}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Prediksi otomatis jika valid
                if is_valid:
                    st.markdown("#### Prediksi Skin Tone")
                    with st.spinner("Menganalisis kulit kamu..."):
                        tensor = preprocess_image(pil_img)
                        probs  = model.predict(tensor, verbose=0)[0]
                        idx    = int(np.argmax(probs))
                        label  = CLASS_NAMES[idx]
                        conf   = float(probs[idx]) * 100

                    st.session_state.prediction    = label
                    st.session_state.probabilities = probs

                    st.markdown(f"""
                    <div class="result-card">
                        <h2>{label.title()}</h2>
                        <p style="font-size:0.95rem;color:#7D4455;">Skin Tone yang terdeteksi</p>
                        <hr style="border-color:#F0A0B8;margin:12px 0;">
                        <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                        <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info("Lihat **Beauty Recommendation** di sidebar untuk saran shade G2G kamu!")
                else:
                    st.warning("Foto belum memenuhi kriteria. Pastikan pencahayaan cukup dan wajah terdeteksi dengan jelas.")

    # ── TAB 2: Camera ──
    with tab2:
        st.markdown("#### Ambil Foto dengan Kamera")
        st.caption("Pastikan pencahayaan cukup dan wajah terlihat jelas.")
        
        # Camera input
        camera_image = st.camera_input("Ambil foto", label_visibility="collapsed")
        
        if camera_image is not None:
            pil_img = Image.open(camera_image)
            st.session_state.camera_image = pil_img
            st.session_state.uploaded_img = pil_img
            
            col_cam, col_cam_info = st.columns([1, 1], gap="large")

            with col_cam:
                st.markdown("#### Hasil Foto")
                st.image(pil_img, use_container_width=True)

            with col_cam_info:
                # Cek kualitas gambar
                st.markdown("#### Validasi Foto")
                quality = check_image_quality(pil_img)
                
                is_valid = (quality["light_status"] == "ok" and quality["face_status"] == "ok")
                
                if is_valid:
                    st.markdown(f"""
                    <div class="validation-pass">
                        <strong>✓</strong> {quality["light_msg"]}<br>
                        <strong>✓</strong> {quality["face_msg"]}<br>
                        <br>
                        <strong style="color:#2E7D32;">Foto valid! Memproses prediksi...</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-process jika valid
                    st.markdown("---")
                    st.markdown("#### Prediksi Skin Tone")
                    with st.spinner("Menganalisis kulit kamu..."):
                        tensor = preprocess_image(pil_img)
                        probs  = model.predict(tensor, verbose=0)[0]
                        idx    = int(np.argmax(probs))
                        label  = CLASS_NAMES[idx]
                        conf   = float(probs[idx]) * 100

                    st.session_state.prediction    = label
                    st.session_state.probabilities = probs

                    st.markdown(f"""
                    <div class="result-card">
                        <h2>{label.title()}</h2>
                        <p style="font-size:0.95rem;color:#7D4455;">Skin Tone yang terdeteksi</p>
                        <hr style="border-color:#F0A0B8;margin:12px 0;">
                        <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                        <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info("Lihat **Beauty Recommendation** di sidebar untuk saran shade G2G kamu!")
                else:
                    st.markdown(f"""
                    <div class="validation-fail">
                        <strong>⚠ Foto belum memenuhi kriteria</strong><br>
                        {quality["light_msg"] if quality["light_status"] == "warn" else ""}
                        {quality["face_msg"] if quality["face_status"] == "warn" else ""}
                        <br><br>
                        <strong>Tips:</strong><br>
                        • Pastikan ruangan cukup terang<br>
                        • Posisikan wajah di tengah frame<br>
                        • Hindari bayangan di wajah
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Klik tombol di atas untuk mengambil foto menggunakan kamera Anda.")

# ═══════════════════════════════════════════════════════════════
# HALAMAN 2 – MODEL INSIGHT
# ═══════════════════════════════════════════════════════════════
elif page == "Model Insight":
    st.markdown("# Model Insight")

    if st.session_state.probabilities is None:
        st.warning("Upload gambar terlebih dahulu di halaman **Upload Image**.")
        st.stop()

    probs = st.session_state.probabilities
    pred  = st.session_state.prediction

    # Confidence score predicted class
    conf = float(probs[CLASS_NAMES.index(pred)]) * 100
    st.metric("Confidence Score", f"{conf:.1f}%", help="Probabilitas kelas yang diprediksi")

    st.markdown("---")
    st.markdown("#### Probabilitas Semua Kelas")

    # Bar chart horizontal
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors  = ["#E8638C" if c == pred else "#F9C6D0" for c in CLASS_NAMES]
    h_bars  = ax.barh(
        [c.title() for c in CLASS_NAMES],
        [p * 100 for p in probs],
        color=colors, height=0.45, edgecolor="white",
    )
    for bar, p in zip(h_bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{p*100:.1f}%", va="center", fontsize=10, color="#3D1A25")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probabilitas (%)", fontsize=10)
    ax.set_facecolor("#FFF8FA")
    fig.patch.set_facecolor("#FFF8FA")
    ax.spines[["top","right"]].set_visible(False)

    legend = [
        mpatches.Patch(color="#E8638C", label="Predicted class"),
        mpatches.Patch(color="#F9C6D0", label="Other classes"),
    ]
    ax.legend(handles=legend, fontsize=9, framealpha=0)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Ringkasan Prediksi")
    rows = []
    for cls, p in zip(CLASS_NAMES, probs):
        rows.append({"Skin Tone": cls.title(), "Probabilitas": f"{p*100:.1f}%",
                     "Status": "Predicted" if cls == pred else ""})
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# HALAMAN 3 – BEAUTY RECOMMENDATION
# ═══════════════════════════════════════════════════════════════
elif page == "Beauty Recommendation":
    st.markdown("# Beauty Recommendation")

    if st.session_state.prediction is None:
        st.warning("Upload gambar terlebih dahulu di halaman **Upload Image**.")
        st.stop()

    pred = st.session_state.prediction
    rec  = G2G_RECOMMENDATION[pred]
    img  = st.session_state.uploaded_img

    col_photo, col_rec = st.columns([1, 1.4], gap="large")

    with col_photo:
        if img:
            st.image(img, caption="Foto kamu", use_container_width=True)

    with col_rec:
        st.markdown(f"### Skin Tone: **{pred.title()}**")
        st.markdown("")

        # Card: G2G Shade
        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">G2G Foundation Shade</p>
            <p style="font-size:1.2rem;font-weight:600;color:#C2185B;margin:0;">{rec['shades']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Card: Skincare
        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">Skincare Recommendation</p>
            <p style="font-size:1.05rem;font-weight:500;color:#3D1A25;margin:0;">{rec['skincare']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Tip
        st.markdown(f"""
        <div class="rec-card" style="border-left-color:#F9C6D0;">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">Beauty Tip</p>
            <p style="font-size:0.95rem;color:#3D1A25;margin:0;">{rec['tip']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Rekomendasi berdasarkan prediksi AI. Lakukan swatching sebelum membeli.")

    # G2G Shade Reference
    st.markdown("---")
    st.markdown("#### Semua Shade G2G")
    shades = [
        ("00", "Allegato",    "#F5D5B0"),
        ("01", "Buttercream", "#F2C89A"),
        ("02", "Praline",     "#D4A278"),
        ("03", "Cookies",     "#C49060"),
        ("04", "Ginger",      "#B07848"),
        ("05", "Cinnamon",    "#8B5E3C"),
    ]
    cols = st.columns(6)
    for col, (code, name, color) in zip(cols, shades):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="width:52px;height:52px;border-radius:50%;background:{color};
                            margin:0 auto 6px;border:2px solid #F0A0B8;
                            box-shadow:0 2px 8px rgba(0,0,0,0.12);"></div>
                <p style="font-size:0.7rem;font-weight:600;color:#C2185B;margin:0;">{code}</p>
                <p style="font-size:0.72rem;color:#7D4455;margin:0;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
