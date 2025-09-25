import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import easyocr
import math
from typing import List, Tuple, Optional, Dict

# Initialize EasyOCR reader
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = image.copy()
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(float) / 255.0
        rgb = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = (img[:, :, c].astype(float) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 3:
        rgb = img[:, :, :3].astype(np.uint8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray_blurred, cleaned

# 🔧 STRICTER CIRCLE DETECTION
def detect_circle(gray_image: np.ndarray, binary_image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    h, w = gray_image.shape[:2]
    min_dim = min(h, w)

    # --- Step 1: Try HoughCircles ---
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dim / 4),        # spread circles apart
        param1=120,
        param2=40,                       # stricter threshold
        minRadius=int(min_dim * 0.25),   # must be at least 25% of canvas
        maxRadius=int(min_dim * 0.48)
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cx, cy, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        return (int(cx), int(cy), int(r))

    # --- Step 2: Contour fallback ---
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_dim * 0.2:   # must be large
            continue
        circle_area = math.pi * (radius ** 2)
        if circle_area <= 0:
            continue
        circularity = area / circle_area
        if circularity < 0.65:       # only keep very round shapes
            continue
        score = circularity * radius
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), int(radius))

    return best_circle

# ---------------------------------------------------------
# (rest of your detect_numbers, detect_hands, scoring, UI, etc.)
# ---------------------------------------------------------
# keep everything else from the previous file unchanged!
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Clock Drawing Test",
        page_icon="🕒",
        layout="wide"
    )

    st.title("🕒 Clock Drawing Test (CDT)")
    st.markdown("### AI-Powered Analysis for Cognitive Assessment")

    if 'circle_detected' not in st.session_state:
        st.session_state.circle_detected = False
    if 'circle_info' not in st.session_state:
        st.session_state.circle_info = None
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'canvas_hash' not in st.session_state:
        st.session_state.canvas_hash = None

    st.markdown("""
    **Instructions:** Please draw a clock showing **10:10**.
    - Step 1: Draw a **big circle** and detect it
    - Step 2: Add numbers 1–12
    - Step 3: Draw hands pointing to 10:10
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Drawing Canvas")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"clock_canvas_{st.session_state.canvas_key}"
        )

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            button_label = "🔍 Analyze Drawing" if st.session_state.circle_detected else "🔵 Detect Circle"
            detect_circle_button = st.button(button_label, type="primary")
        with button_col2:
            if st.button("🗑️ Clear Canvas"):
                st.session_state.canvas_key += 1
                st.session_state.circle_detected = False
                st.session_state.circle_info = None
                st.session_state.canvas_hash = None
                st.rerun()

    with col2:
        st.subheader("Analysis Results")
        if detect_circle_button and canvas_result.image_data is not None:
            image_data = canvas_result.image_data.astype(np.uint8)
            if not np.any(image_data[:, :, 3] > 0):
                st.warning("Please draw something first!")
                return

            gray_blurred, cleaned = preprocess_image(image_data)
            circle_info = detect_circle(gray_blurred, cleaned)

            if circle_info:
                st.session_state.circle_detected = True
                st.session_state.circle_info = circle_info
                cx, cy, radius = circle_info
                st.success(f"✅ Circle detected at ({cx},{cy}) with radius {radius}")
                vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
                cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 2)
                st.image(vis, caption="Detected Circle")
            else:
                st.error("❌ No valid circle detected. Draw a bigger, rounder circle.")

if __name__ == "__main__":
    main()
