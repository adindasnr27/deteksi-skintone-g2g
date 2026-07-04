import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

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
.validation-warning {
    background: #FFF8E1; border-left: 4px solid #F9A825;
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
# LOAD CASCADE CLASSIFIER
# ─────────────────────────────────────────
@st.cache_resource
def get_face_cascade():
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
        
        alt_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                return cv2.CascadeClassifier(path)
        
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        local_path = "/tmp/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, local_path)
        if os.path.exists(local_path):
            return cv2.CascadeClassifier(local_path)
        return None
    except:
        return None

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(pil_img):
    """Preprocessing yang konsisten dengan model training."""
    # Resize ke 224x224
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    
    # Konversi ke array dan normalisasi
    arr = np.array(img, dtype=np.float32)
    
    # Normalisasi ke [0,1] - sesuaikan dengan cara training model Anda
    # Jika model dilatih dengan normalisasi [0,1], gunakan ini
    arr = arr / 255.0
    
    # Jika model dilatih dengan ImageNet normalization, gunakan ini:
    # arr = (arr / 255.0 - 0.5) * 2  # atau menggunakan mean/std ImageNet
    
    return np.expand_dims(arr, axis=0)

def detect_face_skin(pil_img):
    """Deteksi wajah dan ekstrak area kulit dengan lebih presisi."""
    cascade = get_face_cascade()
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = cv_img.shape[:2]
    
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    else:
        faces = []
    
    if len(faces) == 0:
        # Fallback: crop area tengah
        crop_size = min(h, w) // 2
        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        cropped = cv_img[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), False
    
    # Ambil wajah terbesar
    areas = [w * h for (x, y, w, h) in faces]
    largest_idx = np.argmax(areas)
    x, y, w, h = faces[largest_idx]
    
    # Crop area wajah dengan margin yang lebih presisi
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(cv_img.shape[1], x + w + margin_x)
    y2 = min(cv_img.shape[0], y + h + margin_y)
    
    cropped = cv_img[y1:y2, x1:x2]
    
    # Deteksi kulit dengan YCrCb untuk mask yang lebih presisi
    try:
        skin_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2YCrCb)
        # Range warna kulit yang lebih luas
        lower_skin = np.array([0, 130, 75], dtype=np.uint8)
        upper_skin = np.array([255, 175, 130], dtype=np.uint8)
        skin_mask = cv2.inRange(skin_crop, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Jika mask memiliki area yang cukup, gunakan
        if np.sum(skin_mask) > 5000:
            masked = cv2.bitwise_and(cropped, cropped, mask=skin_mask)
            # Crop lagi ke area non-hitam
            coords = cv2.findNonZero(skin_mask)
            if coords is not None:
                x_coords = coords[:,0,0]
                y_coords = coords[:,0,1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                masked = masked[y_min:y_max, x_min:x_max]
            return Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)), True
    except:
        pass
    
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), True

def check_quality(pil_img):
    """Cek kualitas gambar dengan threshold yang lebih fleksibel."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    brightness = float(gray.mean())
    contrast = float(gray.std())
    
    quality_status = "ok"
    quality_msgs = []
    
    if brightness < 40:
        quality_status = "warn"
        quality_msgs.append(f"Cahaya terlalu redup ({brightness:.0f})")
    elif brightness < 70:
        quality_status = "warning"
        quality_msgs.append(f"Cahaya agak redup ({brightness:.0f})")
    elif brightness > 230:
        quality_status = "warn"
        quality_msgs.append(f"Cahaya terlalu terang ({brightness:.0f})")
    elif brightness > 200:
        quality_status = "warning"
        quality_msgs.append(f"Cahaya agak terang ({brightness:.0f})")
    else:
        quality_msgs.append(f"Pencahayaan baik ({brightness:.0f})")
    
    if contrast < 25:
        quality_status = "warning"
        quality_msgs.append("Kontras rendah")
    
    if 50 <= brightness <= 220 and contrast > 20:
        quality_status = "ok"
    
    return quality_status, " | ".join(quality_msgs)

def draw_face_detection(pil_img):
    """Gambar bounding box wajah pada gambar."""
    cascade = get_face_cascade()
    if cascade is None:
        return pil_img
    
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_img, (x, y), (x+w, y+h), (233, 30, 99), 3)
        cv2.rectangle(cv_img, (x+int(w*0.2), y+int(h*0.1)), (x+int(w*0.8), y+int(h*0.3)), (0, 200, 100), 2)
        cv2.rectangle(cv_img, (x+int(w*0.05), y+int(h*0.3)), (x+int(w*0.35), y+int(h*0.7)), (0, 200, 100), 2)
        cv2.rectangle(cv_img, (x+int(w*0.65), y+int(h*0.3)), (x+int(w*0.95), y+int(h*0.7)), (0, 200, 100), 2)
        cv2.rectangle(cv_img, (x+int(w*0.25), y+int(h*0.7)), (x+int(w*0.75), y+int(h*0.9)), (0, 200, 100), 2)
    
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def predict_with_confidence(skin_region):
    """Prediksi dengan confidence threshold."""
    tensor = preprocess_image(skin_region)
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    label = CLASS_NAMES[idx]
    
    # Jika confidence rendah, berikan warning
    if confidence < 60:
        st.warning(f"⚠️ Confidence rendah ({confidence:.1f}%). Hasil mungkin kurang akurat.")
    
    return label, confidence, probs

# ─────────────────────────────────────────
# SESSION STATE & SIDEBAR
# ─────────────────────────────────────────
for key in ["prediction", "probabilities", "uploaded_img", "skin_region", "face_detected"]:
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
    st.info("💡 Pastikan wajah terlihat jelas dan pencahayaan cukup untuk hasil terbaik.")

    if model is None:
        st.error("Model tidak ditemukan. Pastikan `skin_tone_model.h5` ada.")
        st.stop()

    tab1, tab2 = st.tabs(["📤 Upload Foto", "📷 Ambil Foto"])

    with tab1:
        uploaded = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_img = Image.open(uploaded)
            st.session_state.uploaded_img = pil_img
            
            with st.spinner("🔍 Memproses gambar..."):
                skin_region, face_detected = detect_face_skin(pil_img)
                st.session_state.skin_region = skin_region
                st.session_state.face_detected = face_detected
            
            col_img, col_info = st.columns([1, 1], gap="large")
            
            with col_img:
                st.image(pil_img, caption="Foto Asli", use_container_width=True)
                if face_detected:
                    st.image(draw_face_detection(pil_img), caption="Deteksi Wajah", use_container_width=True)
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
                    
                    if status == "ok":
                        st.markdown(f'<div class="validation-pass"><strong>✅</strong> {msg}</div>', unsafe_allow_html=True)
                    elif status == "warning":
                        st.markdown(f'<div class="validation-warning"><strong>⚠️</strong> {msg}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-fail"><strong>❌</strong> {msg}</div>', unsafe_allow_html=True)
                    
                    can_process = (status != "warn")
                    
                    if can_process and skin_region is not None:
                        st.markdown("---")
                        with st.spinner("🔄 Menganalisis kulit..."):
                            label, confidence, probs = predict_with_confidence(skin_region)
                        
                        st.session_state.prediction = label
                        st.session_state.probabilities = probs
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h2>✨ {label.title()}</h2>
                            <p style="color:#7D4455;">Skin Tone terdeteksi</p>
                            <hr style="border-color:#F0A0B8;margin:12px 0;">
                            <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{confidence:.1f}%</p>
                            <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                            <p style="font-size:0.8rem;color:#7D4455;margin-top:8px;">
                                {CLASS_NAMES[0].title()}: {probs[0]*100:.1f}% | 
                                {CLASS_NAMES[1].title()}: {probs[1]*100:.1f}% | 
                                {CLASS_NAMES[2].title()}: {probs[2]*100:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("💖 Lihat **Beauty Recommendation** di sidebar!")

    with tab2:
        st.markdown("#### 📷 Ambil Foto dengan Kamera")
        camera_image = st.camera_input("Ambil foto", label_visibility="collapsed")
        if camera_image:
            pil_img = Image.open(camera_image)
            st.session_state.uploaded_img = pil_img
            
            with st.spinner("🔍 Memproses gambar..."):
                skin_region, face_detected = detect_face_skin(pil_img)
                st.session_state.skin_region = skin_region
                st.session_state.face_detected = face_detected
            
            col_cam, col_cam_info = st.columns([1, 1], gap="large")
            
            with col_cam:
                st.image(pil_img, caption="Hasil Foto", use_container_width=True)
                if face_detected:
                    st.image(draw_face_detection(pil_img), caption="Deteksi Wajah", use_container_width=True)
                if skin_region is not None:
                    st.image(skin_region, caption="Area Kulit", use_container_width=True)
            
            with col_cam_info:
                if not st.session_state.face_detected:
                    st.markdown("""
                    <div class="validation-fail">
                        <strong>⚠️ Wajah tidak terdeteksi</strong><br>
                        Posisikan wajah di tengah frame.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status, msg = check_quality(skin_region)
                    
                    if status == "ok":
                        st.markdown(f'<div class="validation-pass"><strong>✅</strong> {msg}</div>', unsafe_allow_html=True)
                    elif status == "warning":
                        st.markdown(f'<div class="validation-warning"><strong>⚠️</strong> {msg}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-fail"><strong>❌</strong> {msg}</div>', unsafe_allow_html=True)
                    
                    can_process = (status != "warn")
                    
                    if can_process and skin_region is not None:
                        st.markdown("---")
                        with st.spinner("🔄 Menganalisis kulit..."):
                            label, confidence, probs = predict_with_confidence(skin_region)
                        
                        st.session_state.prediction = label
                        st.session_state.probabilities = probs
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h2>✨ {label.title()}</h2>
                            <p style="color:#7D4455;">Skin Tone terdeteksi</p>
                            <hr style="border-color:#F0A0B8;margin:12px 0;">
                            <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{confidence:.1f}%</p>
                            <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
                            <p style="font-size:0.8rem;color:#7D4455;margin-top:8px;">
                                {CLASS_NAMES[0].title()}: {probs[0]*100:.1f}% | 
                                {CLASS_NAMES[1].title()}: {probs[1]*100:.1f}% | 
                                {CLASS_NAMES[2].title()}: {probs[2]*100:.1f}%
                            </p>
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
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E8638C", label="Predicted"),
        Patch(facecolor="#F9C6D0", label="Other")
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
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
