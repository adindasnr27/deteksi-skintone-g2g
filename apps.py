import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
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
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --pink-blush: #F9C6D0; --pink-hot: #E8638C; --pink-deep: #C2185B; --cream: #FFF8FA;
    --text-dark: #3D1A25; --radius: 16px;
}
.stApp { background-color: var(--cream); }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #F9C6D0 0%, #FADADD 60%, #FDE8EF 100%);
    border-right: 1px solid #F0A0B8;
}
[data-testid="stSidebar"] * { color: var(--text-dark) !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--pink-deep) !important; }
p, li, label, div { font-family: 'DM Sans', sans-serif; color: var(--text-dark); }
[data-testid="stMetric"] {
    background: white; border: 1px solid var(--pink-blush); border-radius: var(--radius);
    padding: 18px 22px; box-shadow: 0 2px 12px rgba(232,99,140,0.08);
}
[data-testid="stMetricValue"] {
    color: var(--pink-deep) !important; font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #E8638C, #C2185B); color: white !important;
    border: none; border-radius: 50px; padding: 10px 28px;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
[data-testid="stFileUploader"] {
    background: white; border: 2px dashed var(--pink-blush); border-radius: var(--radius); padding: 16px;
}
.stAlert { border-radius: var(--radius) !important; }
hr { border-color: var(--pink-blush); }
.badge-ok {
    display:inline-block; background:#E8F5E9; color:#2E7D32;
    border-radius:50px; padding:3px 12px; font-size:0.82rem; font-weight:500;
}
.badge-warn {
    display:inline-block; background:#FFF3E0; color:#E65100;
    border-radius:50px; padding:3px 12px; font-size:0.82rem; font-weight:500;
}
.result-card {
    background: linear-gradient(135deg, #FDE8EF, #fff);
    border: 1.5px solid var(--pink-blush); border-radius: var(--radius);
    padding: 24px 28px; margin-top: 16px; box-shadow: 0 4px 20px rgba(232,99,140,0.12);
}
.result-card h2 { margin-bottom: 4px; }
.rec-card {
    background: white; border-left: 4px solid var(--pink-hot);
    border-radius: var(--radius); padding: 20px 24px; margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.validation-pass {
    background: #E8F5E9; border-left: 4px solid #2E7D32;
    padding: 12px 16px; border-radius: 8px; margin: 8px 0;
}
.validation-fail {
    background: #FFF3E0; border-left: 4px solid #E65100;
    padding: 12px 16px; border-radius: 8px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────
CLASS_NAMES = ["dark", "fair", "light"]
IMG_SIZE = (224, 224)
MODEL_PATH = "skin_tone_model.h5"

G2G_RECOMMENDATION = {
    "fair": {
        "shades": "00 Allegato / 01 Buttercream",
        "skincare": "SPF 50+, Brightening Serum, Vitamin C",
        "tip": "Kulit fair rentan sunburn – pastikan pakai sunscreen setiap hari!",
    },
    "light": {
        "shades": "01 Buttercream / 02 Praline",
        "skincare": "SPF 50+, Niacinamide, Hydrating Toner",
        "tip": "Tone kulit light cocok dengan nuansa nude-pink untuk tampilan natural.",
    },
    "dark": {
        "shades": "04 Ginger / 05 Cinnamon",
        "skincare": "Moisturizing Cream, SPF 30+, Shea Butter",
        "tip": "Kulit dark terlihat glowing dengan foundation yang punya undertone warm.",
    },
}

# ─────────────────────────────────────────
# MEDIAPIPE FACE MESH
# ─────────────────────────────────────────
@st.cache_resource
def init_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )

def extract_skin_region(pil_img):
    """Ekstrak area kulit wajah (dahi, pipi, dagu) dengan MediaPipe."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = cv_img.shape[:2]
    
    try:
        face_mesh = init_face_mesh()
        results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            landmarks = []
            for lm in results.multi_face_landmarks[0].landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))
            landmarks = np.array(landmarks)
            
            # Indeks area kulit: dahi, pipi kiri, pipi kanan, dagu
            skin_indices = [10, 108, 67, 109, 10, 234, 227, 116, 117, 118, 119, 120, 121,
                          454, 447, 346, 347, 348, 349, 350, 351, 152, 148, 176, 149, 150, 136, 172]
            skin_points = landmarks[skin_indices]
            
            # Buat mask area kulit
            skin_mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(skin_points.astype(np.int32))
            cv2.fillConvexPoly(skin_mask, hull, 255)
            
            # ROI
            x_min, x_max = min(landmarks[:,0]), max(landmarks[:,0])
            y_min, y_max = min(landmarks[:,1]), max(landmarks[:,1])
            margin_x, margin_y = int((x_max-x_min)*0.15), int((y_max-y_min)*0.15)
            x1, y1 = max(0,x_min-margin_x), max(0,y_min-margin_y)
            x2, y2 = min(w,x_max+margin_x), min(h,y_max+margin_y)
            
            # Aplikasikan mask dan crop
            masked = cv2.bitwise_and(cv_img, cv_img, mask=skin_mask)
            cropped = masked[y1:y2, x1:x2]
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), True, landmarks
    except:
        pass
    
    # Fallback: Haar Cascade
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    
    if len(faces) > 0:
        areas = [w*h for (x,y,w,h) in faces]
        x, y, w, h = faces[np.argmax(areas)]
        margin = int(min(w,h)*0.1)
        x1, y1 = max(0,x-margin), max(0,y-margin)
        x2, y2 = min(w,x+w+margin), min(h,y+h+margin)
        cropped = cv_img[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), False, None
    
    return None, False, None

def draw_landmarks(pil_img, landmarks):
    """Gambar landmark di atas gambar."""
    if landmarks is None:
        return pil_img
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for x, y in landmarks:
        cv2.circle(cv_img, (x, y), 2, (233, 30, 99), -1)
    skin_indices = [10, 108, 67, 109, 10, 234, 227, 116, 117, 118, 119, 120, 121,
                   454, 447, 346, 347, 348, 349, 350, 351, 152, 148, 176, 149, 150, 136, 172]
    hull = cv2.convexHull(landmarks[skin_indices].astype(np.int32))
    cv2.polylines(cv_img, [hull], True, (0, 200, 100), 2)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def check_quality(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    if brightness < 80:
        return "warn", f"Terlalu gelap ({brightness:.0f})"
    elif brightness > 200:
        return "warn", f"Terlalu terang ({brightness:.0f})"
    return "ok", f"Pencahayaan baik ({brightness:.0f})"

# ─────────────────────────────────────────
# SESSION STATE & SIDEBAR
# ─────────────────────────────────────────
for key in ["prediction", "probabilities", "uploaded_img", "skin_region", "landmarks", "face_detected"]:
    if key not in st.session_state:
        st.session_state[key] = None

model = load_model()

with st.sidebar:
    st.markdown("## G2G Shade Finder")
    st.markdown("*AI-powered skin tone detection*")
    st.markdown("---")
    page = st.radio("Navigasi", ["Upload Image", "Model Insight", "Beauty Recommendation"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Final Project · Machine Learning · 2025")

# ═══════════════════════════════════════════════════════════════
# PAGE: UPLOAD IMAGE
# ═══════════════════════════════════════════════════════════════
if page == "Upload Image":
    st.markdown("# Temukan Shade-mu")
    st.markdown("Upload foto wajah atau ambil foto langsung menggunakan kamera.")
    st.info("💡 Pastikan wajah terlihat jelas dan pencahayaan cukup.")

    if model is None:
        st.error("Model tidak ditemukan. Pastikan `skin_tone_model.h5` ada.")
        st.stop()

    tab1, tab2 = st.tabs(["📤 Upload Foto", "📷 Ambil Foto"])

    with tab1:
        uploaded = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_img = Image.open(uploaded)
            st.session_state.uploaded_img = pil_img
            
            with st.spinner("🔍 Mendeteksi titik wajah..."):
                skin_region, landmarks_detected, landmarks = extract_skin_region(pil_img)
                st.session_state.skin_region = skin_region
                st.session_state.landmarks = landmarks
                st.session_state.face_detected = skin_region is not None
            
            col_img, col_info = st.columns([1, 1], gap="large")
            
            with col_img:
                st.image(pil_img, caption="Foto Asli", use_container_width=True)
                if landmarks_detected and landmarks is not None:
                    st.image(draw_landmarks(pil_img, landmarks), caption="Deteksi Titik Wajah", use_container_width=True)
                if skin_region is not None:
                    st.image(skin_region, caption="Area Kulit yang Dianalisis", use_container_width=True)
            
            with col_info:
                if not st.session_state.face_detected:
                    st.markdown("""
                    <div class="validation-fail">
                        <strong>⚠️ Wajah tidak terdeteksi</strong><br>
                        Pastikan wajah terlihat jelas dan menghadap kamera.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status, msg = check_quality(skin_region)
                    is_valid = (status == "ok")
                    
                    if landmarks_detected:
                        st.markdown('<div class="validation-pass"><strong>✅ 468 titik wajah terdeteksi</strong></div>', unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="validation-{"pass" if is_valid else "fail"}"><strong>{"✅" if is_valid else "⚠️"}</strong> {msg}</div>', unsafe_allow_html=True)
                    
                    if is_valid and skin_region is not None:
                        st.markdown("---")
                        with st.spinner("🔄 Menganalisis kulit..."):
                            probs = model.predict(preprocess_image(skin_region), verbose=0)[0]
                            idx = int(np.argmax(probs))
                            label = CLASS_NAMES[idx]
                            conf = float(probs[idx]) * 100
                        
                        st.session_state.prediction = label
                        st.session_state.probabilities = probs
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h2>✨ {label.title()}</h2>
                            <p style="color:#7D4455;">Skin Tone terdeteksi</p>
                            <hr style="border-color:#F0A0B8;margin:12px 0;">
                            <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                            <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("💖 Lihat **Beauty Recommendation** di sidebar!")

    with tab2:
        st.markdown("#### 📷 Ambil Foto dengan Kamera")
        camera_image = st.camera_input("Ambil foto", label_visibility="collapsed")
        if camera_image:
            pil_img = Image.open(camera_image)
            st.session_state.uploaded_img = pil_img
            
            with st.spinner("🔍 Mendeteksi titik wajah..."):
                skin_region, landmarks_detected, landmarks = extract_skin_region(pil_img)
                st.session_state.skin_region = skin_region
                st.session_state.landmarks = landmarks
                st.session_state.face_detected = skin_region is not None
            
            col_cam, col_cam_info = st.columns([1, 1], gap="large")
            
            with col_cam:
                st.image(pil_img, caption="Hasil Foto", use_container_width=True)
                if landmarks_detected and landmarks is not None:
                    st.image(draw_landmarks(pil_img, landmarks), caption="Deteksi Titik Wajah", use_container_width=True)
                if skin_region is not None:
                    st.image(skin_region, caption="Area Kulit", use_container_width=True)
            
            with col_cam_info:
                if not st.session_state.face_detected:
                    st.markdown("""
                    <div class="validation-fail">
                        <strong>⚠️ Wajah tidak terdeteksi</strong><br>
                        Posisikan wajah di tengah frame dan pastikan pencahayaan cukup.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status, msg = check_quality(skin_region)
                    is_valid = (status == "ok")
                    
                    if landmarks_detected:
                        st.markdown('<div class="validation-pass"><strong>✅ 468 titik wajah terdeteksi</strong></div>', unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="validation-{"pass" if is_valid else "fail"}"><strong>{"✅" if is_valid else "⚠️"}</strong> {msg}</div>', unsafe_allow_html=True)
                    
                    if is_valid and skin_region is not None:
                        st.markdown("---")
                        with st.spinner("🔄 Menganalisis kulit..."):
                            probs = model.predict(preprocess_image(skin_region), verbose=0)[0]
                            idx = int(np.argmax(probs))
                            label = CLASS_NAMES[idx]
                            conf = float(probs[idx]) * 100
                        
                        st.session_state.prediction = label
                        st.session_state.probabilities = probs
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h2>✨ {label.title()}</h2>
                            <p style="color:#7D4455;">Skin Tone terdeteksi</p>
                            <hr style="border-color:#F0A0B8;margin:12px 0;">
                            <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                            <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("💖 Lihat **Beauty Recommendation** di sidebar!")

# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHT
# ═══════════════════════════════════════════════════════════════
elif page == "Model Insight":
    st.markdown("# Model Insight")
    
    if st.session_state.probabilities is None:
        st.warning("Upload gambar terlebih dahulu di **Upload Image**.")
        st.stop()
    
    probs, pred = st.session_state.probabilities, st.session_state.prediction
    conf = float(probs[CLASS_NAMES.index(pred)]) * 100
    st.metric("Confidence Score", f"{conf:.1f}%")
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#E8638C" if c == pred else "#F9C6D0" for c in CLASS_NAMES]
    bars = ax.barh([c.title() for c in CLASS_NAMES], [p*100 for p in probs], color=colors, height=0.45)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{p*100:.1f}%", va="center")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probabilitas (%)")
    ax.set_facecolor("#FFF8FA")
    fig.patch.set_facecolor("#FFF8FA")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend([mpatches.Patch(color="#E8638C", label="Predicted"), mpatches.Patch(color="#F9C6D0", label="Other")])
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    import pandas as pd
    rows = [{"Skin Tone": c.title(), "Probabilitas": f"{p*100:.1f}%", "Status": "✅ Predicted" if c == pred else ""} 
            for c, p in zip(CLASS_NAMES, probs)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: BEAUTY RECOMMENDATION
# ═══════════════════════════════════════════════════════════════
elif page == "Beauty Recommendation":
    st.markdown("# Beauty Recommendation")
    
    if st.session_state.prediction is None:
        st.warning("Upload gambar terlebih dahulu di **Upload Image**.")
        st.stop()
    
    pred, rec = st.session_state.prediction, G2G_RECOMMENDATION[st.session_state.prediction]
    skin = st.session_state.skin_region
    
    col_photo, col_rec = st.columns([1, 1.4], gap="large")
    
    with col_photo:
        if skin and st.session_state.face_detected:
            st.image(skin, caption="Area Kulit yang Dianalisis", use_container_width=True)
        elif st.session_state.uploaded_img:
            st.image(st.session_state.uploaded_img, caption="Foto Kamu", use_container_width=True)
    
    with col_rec:
        st.markdown(f"### Skin Tone: **{pred.title()}**")
        
        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;">G2G Foundation Shade</p>
            <p style="font-size:1.2rem;font-weight:600;color:#C2185B;">{rec['shades']}</p>
        </div>
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;">Skincare</p>
            <p style="font-size:1.05rem;font-weight:500;">{rec['skincare']}</p>
        </div>
        <div class="rec-card" style="border-left-color:#F9C6D0;">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;">Beauty Tip</p>
            <p>{rec['tip']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Rekomendasi berdasarkan AI. Lakukan swatching sebelum membeli.")
    
    st.markdown("---")
    st.markdown("#### Semua Shade G2G")
    shades = [("00","Allegato","#F5D5B0"), ("01","Buttercream","#F2C89A"), ("02","Praline","#D4A278"),
              ("03","Cookies","#C49060"), ("04","Ginger","#B07848"), ("05","Cinnamon","#8B5E3C")]
    cols = st.columns(6)
    for col, (code, name, color) in zip(cols, shades):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="width:52px;height:52px;border-radius:50%;background:{color};
                            margin:0 auto 6px;border:2px solid #F0A0B8;box-shadow:0 2px 8px rgba(0,0,0,0.12);"></div>
                <p style="font-size:0.7rem;font-weight:600;color:#C2185B;margin:0;">{code}</p>
                <p style="font-size:0.72rem;color:#7D4455;margin:0;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
