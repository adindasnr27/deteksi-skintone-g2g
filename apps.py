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
import mediapipe as mp

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

/* Face crop result */
.face-crop-container {
    background: white;
    border: 2px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 16px;
    margin: 12px 0;
    text-align: center;
}

/* Landmark visualization */
.landmark-container {
    background: white;
    border: 2px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 16px;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────
CLASS_NAMES = ["dark", "fair", "light"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "skin_tone_model.h5"

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
# INITIALIZE MEDIAPIPE
# ─────────────────────────────────────────
@st.cache_resource
def init_face_mesh():
    """Initialize MediaPipe Face Mesh."""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )

# ─────────────────────────────────────────
# HELPER: Load model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────
# HELPER: Face Landmark Detection dengan MediaPipe
# ─────────────────────────────────────────
def get_face_skin_mask(pil_img: Image.Image):
    """
    Deteksi 468 titik wajah menggunakan MediaPipe dan buat mask area kulit wajah.
    Returns: (skin_mask, face_roi, landmarks_visible)
    """
    # Konversi ke OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = cv_img.shape[:2]
    
    # Inisialisasi mask
    skin_mask = np.zeros((h, w), dtype=np.uint8)
    face_roi = None
    landmarks_visible = False
    
    try:
        face_mesh = init_face_mesh()
        results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            landmarks_visible = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Ambil koordinat landmark
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            landmarks = np.array(landmarks)
            
            # Area kulit wajah yang penting:
            # - Dahi: indeks 10, 108, 67, 109, 10 (area atas)
            # - Pipi kiri: indeks 234, 227, 116, 117, 118, 119, 120, 121
            # - Pipi kanan: indeks 454, 447, 346, 347, 348, 349, 350, 351
            # - Dagu: indeks 152, 148, 176, 149, 150, 136, 172
            
            skin_indices = [
                # Dahi (atas)
                10, 108, 67, 109, 10,
                # Pipi kiri
                234, 227, 116, 117, 118, 119, 120, 121,
                # Pipi kanan
                454, 447, 346, 347, 348, 349, 350, 351,
                # Dagu
                152, 148, 176, 149, 150, 136, 172
            ]
            
            # Ambil titik-titik kulit
            skin_points = landmarks[skin_indices]
            
            # Buat convex hull untuk area kulit
            hull = cv2.convexHull(skin_points.astype(np.int32))
            cv2.fillConvexPoly(skin_mask, hull, 255)
            
            # Buat ROI dari semua landmark
            x_min = min(landmarks[:, 0])
            x_max = max(landmarks[:, 0])
            y_min = min(landmarks[:, 1])
            y_max = max(landmarks[:, 1])
            
            # Tambahkan margin 15%
            margin_x = int((x_max - x_min) * 0.15)
            margin_y = int((y_max - y_min) * 0.15)
            
            x1 = max(0, x_min - margin_x)
            y1 = max(0, y_min - margin_y)
            x2 = min(w, x_max + margin_x)
            y2 = min(h, y_max + margin_y)
            
            face_roi = (x1, y1, x2, y2)
            
            return skin_mask, face_roi, landmarks_visible, landmarks
            
    except Exception as e:
        st.warning(f"⚠️ MediaPipe error: {e}. Menggunakan deteksi wajah dasar.")
    
    # Fallback: Haar Cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) > 0:
        # Ambil wajah terbesar
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        x, y, w, h = faces[largest_idx]
        
        # Crop area wajah dengan margin
        margin = int(min(w, h) * 0.1)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(cv_img.shape[1], x + w + margin)
        y2 = min(cv_img.shape[0], y + h + margin)
        
        # Buat mask sederhana (area wajah)
        skin_mask[y1:y2, x1:x2] = 255
        face_roi = (x1, y1, x2, y2)
        
        return skin_mask, face_roi, False, None
    
    return None, None, False, None

def extract_skin_region(pil_img: Image.Image):
    """
    Ekstrak hanya area kulit wajah (dahi, pipi, dagu) dari gambar.
    Returns: (skin_region, mask, landmarks, landmarks_visible)
    """
    skin_mask, face_roi, landmarks_visible, landmarks = get_face_skin_mask(pil_img)
    
    if skin_mask is None or face_roi is None:
        return None, None, None, False
    
    # Konversi PIL ke OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Aplikasikan mask ke gambar
    masked_img = cv2.bitwise_and(cv_img, cv_img, mask=skin_mask)
    
    # Crop ke area wajah
    x1, y1, x2, y2 = face_roi
    cropped = masked_img[y1:y2, x1:x2]
    
    # Konversi kembali ke PIL
    skin_region = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    return skin_region, skin_mask, landmarks, landmarks_visible

def draw_landmarks(pil_img: Image.Image, landmarks):
    """Gambar titik-titik landmark di atas gambar."""
    if landmarks is None:
        return pil_img
    
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Gambar titik-titik landmark (warna pink)
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(cv_img, (x, y), 2, (233, 30, 99), -1)
    
    # Gambar area kulit (convex hull)
    skin_indices = [
        10, 108, 67, 109, 10,  # dahi
        234, 227, 116, 117, 118, 119, 120, 121,  # pipi kiri
        454, 447, 346, 347, 348, 349, 350, 351,  # pipi kanan
        152, 148, 176, 149, 150, 136, 172  # dagu
    ]
    skin_points = landmarks[skin_indices]
    hull = cv2.convexHull(skin_points.astype(np.int32))
    cv2.polylines(cv_img, [hull], True, (0, 200, 100), 2)
    
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────
# HELPER: Preprocessing gambar
# ─────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ─────────────────────────────────────────
# HELPER: Cek kualitas gambar
# ─────────────────────────────────────────
def check_image_quality(pil_img: Image.Image):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    if brightness < 80:
        light_status, light_msg = "warn", f"Terlalu gelap (brightness {brightness:.0f})"
    elif brightness > 200:
        light_status, light_msg = "warn", f"Terlalu terang (brightness {brightness:.0f})"
    else:
        light_status, light_msg = "ok", f"Pencahayaan baik (brightness {brightness:.0f})"

    return {
        "light_status": light_status,
        "light_msg": light_msg,
    }

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
    st.session_state.prediction = None
    st.session_state.probabilities = None
    st.session_state.uploaded_img = None
    st.session_state.skin_region = None
    st.session_state.landmarks_visible = False
    st.session_state.face_detected = False

model = load_model()

# ═══════════════════════════════════════════════════════════════
# HALAMAN 1 – UPLOAD IMAGE
# ═══════════════════════════════════════════════════════════════
if page == "Upload Image":
    st.markdown("# Temukan Shade-mu")
    st.markdown("Upload foto wajah atau ambil foto langsung menggunakan kamera.")
    st.info("💡 **Tips:** Pastikan wajah terlihat jelas, pencahayaan cukup, dan tidak ada aksesoris yang menutupi wajah.")

    if model is None:
        st.error("Model belum ditemukan. Pastikan `skin_tone_model.h5` ada di repo.")
        st.stop()

    tab1, tab2 = st.tabs(["📤 Upload Foto", "📷 Ambil Foto"])

    # ── TAB 1: Upload ──
    with tab1:
        uploaded = st.file_uploader("Upload foto wajah (JPG / PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded:
            pil_img = Image.open(uploaded)
            st.session_state.uploaded_img = pil_img
            
            # Ekstrak area kulit wajah
            with st.spinner("🔍 Mendeteksi titik-titik wajah (dahi, pipi, dagu)..."):
                skin_region, skin_mask, landmarks, landmarks_visible = extract_skin_region(pil_img)
                st.session_state.skin_region = skin_region
                st.session_state.landmarks_visible = landmarks_visible
                st.session_state.face_detected = skin_region is not None
            
            col_img, col_info = st.columns([1, 1], gap="large")

            with col_img:
                st.markdown("#### 📸 Foto Asli")
                st.image(pil_img, use_container_width=True)
                
                if landmarks_visible and landmarks is not None:
                    st.markdown("#### 🎯 Deteksi Titik Wajah (468 titik)")
                    landmark_img = draw_landmarks(pil_img, landmarks)
                    st.image(landmark_img, use_container_width=True)
                    st.caption("✅ Area dahi, pipi, dan dagu terdeteksi")
                
                if skin_region is not None:
                    st.markdown("#### 🎨 Area Kulit yang Dianalisis")
                    st.image(skin_region, use_container_width=True)
                    st.caption("✅ Hanya area kulit wajah yang diproses (bukan background/kerudung)")

            with col_info:
                if not st.session_state.face_detected:
                    st.markdown(f"""
                    <div class="validation-fail">
                        <strong>⚠️ Wajah tidak terdeteksi</strong><br>
                        Mohon upload foto dengan wajah yang jelas.
                        <br><br>
                        <strong>Tips:</strong><br>
                        • Gunakan foto close-up<br>
                        • Pastikan wajah menghadap kamera<br>
                        • Hindari aksesoris yang menutupi wajah<br>
                        • Pastikan pencahayaan cukup
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    quality = check_image_quality(skin_region)
                    is_valid = (quality["light_status"] == "ok")
                    
                    st.markdown("#### ✨ Validasi Foto")
                    
                    if landmarks_visible:
                        st.markdown(f"""
                        <div class="validation-pass">
                            <strong>✅ 468 titik wajah terdeteksi (MediaPipe)</strong><br>
                            <strong>✅ Dahi, pipi, dan dagu teridentifikasi</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="validation-warn">
                            <strong>⚠️ Menggunakan deteksi wajah dasar (Haar Cascade)</strong><br>
                            Untuk hasil terbaik, pastikan MediaPipe terinstall dengan baik.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if is_valid:
                        st.markdown(f"""
                        <div class="validation-pass">
                            <strong>✅</strong> {quality["light_msg"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="validation-fail">
                            <strong>⚠️</strong> {quality["light_msg"]}
                            <br><br>
                            <strong>Tips:</strong><br>
                            • Cari pencahayaan yang lebih baik<br>
                            • Hindari bayangan di wajah
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    if is_valid and skin_region is not None:
                        st.markdown("#### 💄 Analisis Skin Tone")
                        with st.spinner("🔄 Menganalisis kulit wajah kamu..."):
                            tensor = preprocess_image(skin_region)
                            probs = model.predict(tensor, verbose=0)[0]
                            idx = int(np.argmax(probs))
                            label = CLASS_NAMES[idx]
                            conf = float(probs[idx]) * 100

                        st.session_state.prediction = label
                        st.session_state.probabilities = probs

                        st.markdown(f"""
                        <div class="result-card">
                            <h2>✨ {label.title()}</h2>
                            <p style="font-size:0.95rem;color:#7D4455;">Skin Tone yang terdeteksi</p>
                            <hr style="border-color:#F0A0B8;margin:12px 0;">
                            <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                            <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.success("💖 Lihat **Beauty Recommendation** di sidebar untuk saran shade G2G kamu!")
                    else:
                        st.warning("⚠️ Kualitas gambar kurang baik. Silakan upload foto dengan pencahayaan yang lebih baik.")

    # ── TAB 2: Camera ──
    with tab2:
        st.markdown("#### 📷 Ambil Foto dengan Kamera")
        st.caption("Pastikan wajah terlihat jelas dan pencahayaan cukup.")
        
        camera_image = st.camera_input("📸 Ambil foto", label_visibility="collapsed")
        
        if camera_image is not None:
            pil_img = Image.open(camera_image)
            st.session_state.uploaded_img = pil_img
            
            with st.spinner("🔍 Mendeteksi titik-titik wajah..."):
                skin_region, skin_mask, landmarks, landmarks_visible = extract_skin_region(pil_img)
                st.session_state.skin_region = skin_region
                st.session_state.landmarks_visible = landmarks_visible
                st.session_state.face_detected = skin_region is not None
            
            col_cam, col_cam_info = st.columns([1, 1], gap="large")

            with col_cam:
                st.markdown("#### 📸 Hasil Foto")
                st.image(pil_img, use_container_width=True)
                
                if landmarks_visible and landmarks is not None:
                    st.markdown("#### 🎯 Deteksi Titik Wajah")
                    landmark_img = draw_landmarks(pil_img, landmarks)
                    st.image(landmark_img, use_container_width=True)
                
                if skin_region is not None:
                    st.markdown("#### 🎨 Area Kulit yang Dianalisis")
                    st.image(skin_region, use_container_width=True)

            with col_cam_info:
                if not st.session_state.face_detected:
                    st.markdown(f"""
                    <div class="validation-fail">
                        <strong>⚠️ Wajah tidak terdeteksi</strong><br>
                        Mohon ambil foto dengan wajah yang jelas.
                        <br><br>
                        <strong>Tips:</strong><br>
                        • Posisikan wajah di tengah frame<br>
                        • Pastikan wajah menghadap kamera<br>
                        • Ambil foto dari jarak dekat<br>
                        • Pastikan pencahayaan cukup
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    quality = check_image_quality(skin_region)
                    is_valid = (quality["light_status"] == "ok")
                    
                    st.markdown("#### ✨ Validasi Foto")
                    
                    if landmarks_visible:
                        st.markdown(f"""
                        <div class="validation-pass">
                            <strong>✅ 468 titik wajah terdeteksi</strong><br>
                            <strong>✅ Dahi, pipi, dan dagu teridentifikasi</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if is_valid:
                        st.markdown(f"""
                        <div class="validation-pass">
                            <strong>✅</strong> {quality["light_msg"]}
                            <br><br>
                            <strong style="color:#2E7D32;">🎉 Foto valid! Memproses prediksi...</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="validation-fail">
                            <strong>⚠️</strong> {quality["light_msg"]}
                            <br><br>
                            <strong>Tips:</strong><br>
                            • Cari pencahayaan yang lebih baik<br>
                            • Hindari bayangan di wajah
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    if is_valid and skin_region is not None:
                        st.markdown("#### 💄 Analisis Skin Tone")
                        with st.spinner("🔄 Menganalisis kulit wajah kamu..."):
                            tensor = preprocess_image(skin_region)
                            probs = model.predict(tensor, verbose=0)[0]
                            idx = int(np.argmax(probs))
                            label = CLASS_NAMES[idx]
                            conf = float(probs[idx]) * 100

                        st.session_state.prediction = label
                        st.session_state.probabilities = probs

                        st.markdown(f"""
                        <div class="result-card">
                            <h2>✨ {label.title()}</h2>
                            <p style="font-size:0.95rem;color:#7D4455;">Skin Tone yang terdeteksi</p>
                            <hr style="border-color:#F0A0B8;margin:12px 0;">
                            <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                            <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.success("💖 Lihat **Beauty Recommendation** di sidebar untuk saran shade G2G kamu!")
                    else:
                        st.warning("⚠️ Kualitas gambar kurang baik. Silakan ambil foto dengan pencahayaan yang lebih baik.")
        else:
            st.info("📸 Klik tombol kamera di atas untuk mengambil foto.")

# ═══════════════════════════════════════════════════════════════
# HALAMAN 2 – MODEL INSIGHT
# ═══════════════════════════════════════════════════════════════
elif page == "Model Insight":
    st.markdown("# Model Insight")

    if st.session_state.probabilities is None:
        st.warning("Upload gambar terlebih dahulu di halaman **Upload Image**.")
        st.stop()

    probs = st.session_state.probabilities
    pred = st.session_state.prediction

    conf = float(probs[CLASS_NAMES.index(pred)]) * 100
    st.metric("Confidence Score", f"{conf:.1f}%", help="Probabilitas kelas yang diprediksi")

    st.markdown("---")
    st.markdown("#### Probabilitas Semua Kelas")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#E8638C" if c == pred else "#F9C6D0" for c in CLASS_NAMES]
    h_bars = ax.barh(
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
    rec = G2G_RECOMMENDATION[pred]
    img = st.session_state.uploaded_img
    skin = st.session_state.skin_region

    col_photo, col_rec = st.columns([1, 1.4], gap="large")

    with col_photo:
        if skin and st.session_state.face_detected:
            st.markdown("#### 🎨 Area Kulit yang Dianalisis")
            st.image(skin, use_container_width=True)
            st.caption("Area dahi, pipi, dan dagu yang digunakan untuk prediksi")
        elif img:
            st.image(img, caption="Foto kamu", use_container_width=True)

    with col_rec:
        st.markdown(f"### Skin Tone: **{pred.title()}**")
        st.markdown("")

        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">G2G Foundation Shade</p>
            <p style="font-size:1.2rem;font-weight:600;color:#C2185B;margin:0;">{rec['shades']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">Skincare Recommendation</p>
            <p style="font-size:1.05rem;font-weight:500;color:#3D1A25;margin:0;">{rec['skincare']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-card" style="border-left-color:#F9C6D0;">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">Beauty Tip</p>
            <p style="font-size:0.95rem;color:#3D1A25;margin:0;">{rec['tip']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Rekomendasi berdasarkan prediksi AI. Lakukan swatching sebelum membeli.")

    st.markdown("---")
    st.markdown("#### Semua Shade G2G")
    shades = [
        ("00", "Allegato", "#F5D5B0"),
        ("01", "Buttercream", "#F2C89A"),
        ("02", "Praline", "#D4A278"),
        ("03", "Cookies", "#C49060"),
        ("04", "Ginger", "#B07848"),
        ("05", "Cinnamon", "#8B5E3C"),
    ]
    cols = st.columns(6)
    for col, (code, name, color) in zip(cols, shades):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="width:52px;height:52px;border-radius:50%;background:{color};
                            margin:0 auto 6px;border:2px solid #F0A0B8;
                            box-shadow:0 2px 8px rgba(0,0,0,0.12);"></div>
                <p
