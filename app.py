import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Mask Detector", page_icon="😷", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detection_model.h5")

def predict(model, img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    return float(probs[0]), float(probs[1])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("😷 Face Mask Detector")
st.markdown("Upload a face image — result appears instantly. **Refresh the page** to run another.")
st.divider()

model = load_model()

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if file is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

# ── Run detection exactly once ────────────────────────────────────────────────
img = Image.open(file)
with st.spinner("Analysing…"):
    with_mask_prob, without_mask_prob = predict(model, img)

wearing = with_mask_prob >= without_mask_prob
label   = "Mask On ✅"  if wearing else "No Mask ⚠️"
emoji   = "😷"          if wearing else "🙅"
color   = "green"       if wearing else "red"
conf    = with_mask_prob if wearing else without_mask_prob

# ── Display result ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.4])
with col1:
    st.image(img, use_container_width=True, caption=file.name)
with col2:
    st.markdown(f"## {emoji} {label}")
    st.markdown(f":{color}[**Confidence: {conf:.1%}**]")
    st.divider()
    st.progress(with_mask_prob,    text=f"With Mask  : {with_mask_prob:.1%}")
    st.progress(without_mask_prob, text=f"No Mask    : {without_mask_prob:.1%}")

st.divider()
st.info("🔄 Refresh the page to upload a new image.")

# ── Block any further interaction ─────────────────────────────────────────────
st.stop()