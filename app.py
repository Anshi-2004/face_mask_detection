# =============================================================
# app.py — Face Mask Detection: Streamlit GUI (Light Theme + Dots)
# Run with:  streamlit run app.py
# =============================================================

import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import io

# ─── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Light Theme CSS with Floating Dots ──────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #e4e3f3 0%, #ebebf5 40%, #fbe8df 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.6);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #1e2022;
    }

    /* ── Hero Title ── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(134, 133, 207, 0.12);
        border: 1px solid rgba(134, 133, 207, 0.3);
        border-radius: 50px;
        padding: 0.4rem 1.2rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: #8685cf;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #8685cf 0%, #6866b8 50%, #fcae86 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.6rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Glass Cards (Light) ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 1);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(134, 133, 207, 0.08);
    }
    .glass-card:hover {
        border-color: rgba(134, 133, 207, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(134, 133, 207, 0.15);
    }

    .card-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
    .card-value {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .card-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Card accents */
    .card-purple .card-value { color: #8685cf; }
    .card-blue   .card-value { color: #a2a1da; }
    .card-amber  .card-value { color: #fed1b7; text-shadow: 0 1px 2px rgba(254, 209, 183, 0.4); }
    .card-green  .card-value { color: #059669; }
    .card-red    .card-value { color: #dc2626; }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
        border-bottom: 2px solid rgba(134, 133, 207, 0.2);
    }

    /* ── Result Badges ── */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-mask {
        background: rgba(5, 150, 105, 0.1);
        color: #059669;
        border: 1px solid rgba(5, 150, 105, 0.2);
    }
    .badge-nomask {
        background: rgba(220, 38, 38, 0.08);
        color: #dc2626;
        border: 1px solid rgba(220, 38, 38, 0.2);
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(134, 133, 207, 0.15);
        border-left: 3px solid #8685cf;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.7;
        backdrop-filter: blur(8px);
    }

    /* ── Sidebar Info Card ── */
    .sidebar-info {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(134, 133, 207, 0.12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #1e2022;
        font-size: 0.88rem;
        line-height: 1.8;
    }
    .sidebar-info b { color: #8685cf; }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(134, 133, 207, 0.2) 50%,
            transparent 100%);
        margin: 1.5rem 0;
    }

    /* ── Face Result Card ── */
    .face-result-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(134, 133, 207, 0.12);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 10px rgba(134, 133, 207, 0.05);
    }
    .face-result-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .face-result-stats {
        color: #64748b;
        font-size: 0.85rem;
    }
    .face-result-stats b { color: #1e293b; }

    /* ── Tech Table ── */
    .tech-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(134, 133, 207, 0.15);
        background: rgba(255, 255, 255, 0.8);
    }
    .tech-table th {
        background: rgba(134, 133, 207, 0.08);
        color: #8685cf;
        padding: 0.7rem 1rem;
        text-align: left;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tech-table td {
        padding: 0.65rem 1rem;
        color: #1e2022;
        font-size: 0.88rem;
        border-top: 1px solid rgba(134, 133, 207, 0.06);
    }
    .tech-table tr:hover td {
        background: rgba(134, 133, 207, 0.04);
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(134, 133, 207, 0.15);
        backdrop-filter: blur(8px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        color: #64748b;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8685cf, #6866b8);
        color: #ffffff !important;
        border: none;
        box-shadow: 0 4px 12px rgba(134, 133, 207, 0.25);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"]    { display: none; }

    /* ── Architecture Block ── */
    .arch-block {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(134, 133, 207, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        color: #8685cf;
        line-height: 1.9;
        white-space: pre;
        overflow-x: auto;
        backdrop-filter: blur(8px);
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #8685cf, #ffb38a) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(134, 133, 207, 0.35) !important;
    }

</style>
""", unsafe_allow_html=True)

# ─── Lazy-load heavy dependencies ────────────────────────────
@st.cache_resource(show_spinner="🧠 Loading AI model...")
def load_models():
    """Load face detector and mask classifier once, cache for session."""
    from tensorflow.keras.models import load_model as keras_load

    FACE_PROTO   = "face_detector/deploy.prototxt"
    FACE_WEIGHTS = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    MODEL_PATH   = "mask_detector.keras"

    errors = []
    faceNet = None
    maskNet = None

    if (os.path.exists(FACE_PROTO) and os.path.getsize(FACE_PROTO) > 0
            and os.path.exists(FACE_WEIGHTS) and os.path.getsize(FACE_WEIGHTS) > 0):
        faceNet = cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)
    else:
        errors.append("Face detector files missing or empty in `face_detector/` folder.")

    if os.path.exists(MODEL_PATH):
        maskNet = keras_load(MODEL_PATH)
    else:
        errors.append("`mask_detector.keras` not found. Run `train.py` first.")

    return faceNet, maskNet, errors


def preprocess_face(face_bgr):
    """Resize + preprocess a face crop for the model."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    return np.array([face], dtype="float32")


def detect_faces_and_predict(image_bgr, faceNet, maskNet, min_conf=0.5):
    """
    Run face detection + mask classification on a BGR image.
    Returns annotated image and list of result dicts.
    """
    (h, w) = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < min_conf:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        (x1, y1) = (max(0, x1), max(0, y1))
        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

        face = image_bgr[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_input = preprocess_face(face)
        (maskProb, noMaskProb) = maskNet.predict(face_input, verbose=0)[0]

        label      = "Mask" if maskProb > noMaskProb else "No Mask"
        conf_score = max(maskProb, noMaskProb) * 100
        border_color = (34, 197, 94) if label == "Mask" else (239, 68, 68)

        # Bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), border_color, 2)

        # Label background
        overlay = image_bgr.copy()
        text = f"{label}: {conf_score:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay,
                      (x1, y1 - th - 14),
                      (x1 + tw + 12, y1),
                      border_color, -1)
        cv2.addWeighted(overlay, 0.85, image_bgr, 0.15, 0, image_bgr)

        cv2.putText(image_bgr, text,
                    (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        results.append({
            "label":      label,
            "confidence": conf_score,
            "mask_prob":  float(maskProb) * 100,
            "nomask_prob":float(noMaskProb) * 100,
            "face_num":   len(results) + 1
        })

    return image_bgr, results


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size: 2.8rem;">😷</span>
        <div style="font-size: 1.1rem; font-weight: 800; color: #8685cf;
                    margin-top: 0.3rem; letter-spacing: 1px;">MASK DETECTOR</div>
        <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.2rem;
                    letter-spacing: 2px; text-transform: uppercase;">AI-Powered</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ Detection Settings")

    min_conf_thresh = st.slider(
        "Min Face Detection Threshold", 0.1, 1.0, 0.7, 0.05,
        help="Minimum confidence required to accept a detected face. "
             "Higher = stricter (fewer false positives). "
             "Lower = more sensitive (may detect non-faces)."
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📊 Model Info")
    st.markdown("""
    <div class='sidebar-info'>
        <b>Architecture:</b> MobileNetV2<br>
        <b>Training:</b> Transfer Learning<br>
        <b>Input Size:</b> 224 × 224<br>
        <b>Classes:</b> Mask / No Mask<br>
        <b>Framework:</b> TensorFlow + Keras
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 🎨 Detection Legend")
    st.markdown("""
    <div style="padding: 0 0.2rem;">
        <div style="display:flex; align-items:center; gap: 0.6rem; margin-bottom: 0.5rem;">
            <span style="width:14px; height:14px; border-radius:4px; background:#059669; display:inline-block;"></span>
            <span style="color:#334155; font-size:0.88rem; font-weight:500;">Mask Detected</span>
        </div>
        <div style="display:flex; align-items:center; gap: 0.6rem;">
            <span style="width:14px; height:14px; border-radius:4px; background:#dc2626; display:inline-block;"></span>
            <span style="color:#334155; font-size:0.88rem; font-weight:500;">No Mask Detected</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0;">
        <span style="color: #94a3b8; font-size: 0.72rem;">
            Built with Streamlit · TensorFlow · OpenCV
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Page Hero ──────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🛡️ Deep Learning Powered</div>
    <div class="hero-title">Face Mask Detection</div>
    <div class="hero-subtitle">
        Real-time mask detection using MobileNetV2 transfer learning with
        OpenCV face detection pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
faceNet, maskNet, load_errors = load_models()

# Show model loading errors
if load_errors:
    for err in load_errors:
        st.warning(f"⚠️ {err}")
    st.info("📌 **Setup Checklist:**\n"
            "1. Run `python train.py` to train and save the model.\n"
            "2. Download face detector files (see README).\n"
            "3. Reload this page.")

# ─── Metric Cards ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="glass-card card-purple">
        <div class="card-icon">🧠</div>
        <div class="card-value">MobileNetV2</div>
        <div class="card-label">Architecture</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card card-blue">
        <div class="card-icon">📐</div>
        <div class="card-value">224×224</div>
        <div class="card-label">Input Resolution</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card card-amber">
        <div class="card-icon">🏷️</div>
        <div class="card-value">2 Classes</div>
        <div class="card-label">Mask / No Mask</div>
    </div>""", unsafe_allow_html=True)

with col4:
    if maskNet:
        st.markdown("""
        <div class="glass-card card-green">
            <div class="card-icon">✅</div>
            <div class="card-value">Loaded</div>
            <div class="card-label">Model Status</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card card-red">
            <div class="card-icon">❌</div>
            <div class="card-value">Missing</div>
            <div class="card-label">Model Status</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🖼️  Image Upload", "📷  Webcam Capture"])

# ═══════════════════════════════════════════════
# TAB 1 — Image Upload
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
        <span>🖼️</span> Upload an Image for Detection
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo containing one or more faces."
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_result = st.columns(2)

        with col_orig:
            st.markdown("**📸 Original Image**")
            st.image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

        if faceNet and maskNet:
            with st.spinner("🔍 Detecting faces and predicting..."):
                annotated, results = detect_faces_and_predict(
                    image_bgr.copy(), faceNet, maskNet, min_conf_thresh
                )

            with col_result:
                st.markdown("**🎯 Detection Result**")
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            if not results:
                st.warning("🔍 No faces detected. Try a clearer, front-facing photo or lower the detection threshold.")
            else:
                st.markdown(f"""
                <div class="section-header">
                    <span>🎯</span> Detected {len(results)} Face(s)
                </div>
                """, unsafe_allow_html=True)

                mask_count   = sum(1 for r in results if r["label"] == "Mask")
                nomask_count = len(results) - mask_count

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="glass-card card-blue">
                        <div class="card-value">{len(results)}</div>
                        <div class="card-label">Total Faces</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="glass-card card-green">
                        <div class="card-value">{mask_count}</div>
                        <div class="card-label">Wearing Mask</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="glass-card card-red">
                        <div class="card-value">{nomask_count}</div>
                        <div class="card-label">No Mask</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                for r in results:
                    badge_class = "badge-mask" if r["label"] == "Mask" else "badge-nomask"
                    badge_icon = "✅" if r["label"] == "Mask" else "❌"

                    st.markdown(f"""
                    <div class="face-result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                            <div class="face-result-title">👤 Face #{r["face_num"]}</div>
                            <span class="result-badge {badge_class}">{badge_icon} {r["label"]}</span>
                        </div>
                        <div class="face-result-stats" style="margin-top:0.5rem;">
                            Confidence: <b>{r["confidence"]:.1f}%</b> &nbsp;│&nbsp;
                            Mask: <b style="color:#059669;">{r["mask_prob"]:.1f}%</b> &nbsp;│&nbsp;
                            No Mask: <b style="color:#dc2626;">{r["nomask_prob"]:.1f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    cols = st.columns(2)
                    cols[0].progress(int(r["mask_prob"]),    text=f"Mask: {r['mask_prob']:.1f}%")
                    cols[1].progress(int(r["nomask_prob"]),  text=f"No Mask: {r['nomask_prob']:.1f}%")

                st.markdown("")

                result_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="mask_detection_result.png",
                    mime="image/png"
                )
        else:
            with col_result:
                st.error("Models not loaded. Check the sidebar warnings.")


# ═══════════════════════════════════════════════
# TAB 2 — Webcam
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
        <span>📷</span> Real-Time Webcam Detection
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <b>How it works:</b> Click <b>Start</b> to capture a frame from your webcam.
        Each captured frame is processed through the AI pipeline. For continuous
        real-time detection at full speed, use <code>python detect.py</code> in your terminal.
    </div>
    """, unsafe_allow_html=True)

    run_webcam = st.checkbox("🎥 Enable Webcam", value=False, key="webcam_toggle")

    if run_webcam:
        cam_image = st.camera_input("Point your camera at a face")

        if cam_image is not None and faceNet and maskNet:
            file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
            frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("🔍 Analysing frame..."):
                annotated, results = detect_faces_and_predict(
                    frame_bgr.copy(), faceNet, maskNet, min_conf_thresh
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**📸 Original Frame**")
                st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col_b:
                st.markdown("**🎯 Annotated Frame**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            if results:
                mask_c   = sum(1 for r in results if r["label"] == "Mask")
                nomask_c = len(results) - mask_c
                st.success(f"✅ Detected {len(results)} face(s): "
                           f"**{mask_c} with mask**, **{nomask_c} without mask**")
                for r in results:
                    badge = "✅ Mask" if r["label"] == "Mask" else "❌ No Mask"
                    st.write(f"Face #{r['face_num']}: {badge} — {r['confidence']:.1f}% confidence")
            else:
                st.warning("No faces detected in this frame.")

        elif not faceNet or not maskNet:
            st.error("Models not loaded. Please check setup.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <span>🖥️</span> Command-Line Detection
    </div>
    """, unsafe_allow_html=True)

    st.code("python detect.py", language="bash")
    st.markdown("Opens an OpenCV window with **live bounding boxes**. Press **Q** to quit.")
    st.code("python detect.py --image path/to/photo.jpg", language="bash")
    st.markdown("Detect masks in a single image file.")


