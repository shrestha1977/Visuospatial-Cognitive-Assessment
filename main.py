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
    """
    Take canvas image (likely RGBA) and return:
      - gray_blurred: a blurred grayscale image (uint8) suitable for HoughCircles/HoughLines
      - cleaned: a binary image (uint8) with strokes as white (255) on black (0)
    """
    img = image.copy()
    # If RGBA, composite on white background to preserve strokes
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(float) / 255.0
        rgb = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = (img[:, :, c].astype(float) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 3:
        rgb = img[:, :, :3].astype(np.uint8)
    else:
        # single channel
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Ensure it's RGB order for cv2 operations (Streamlit is RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise for Hough/Canny
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Binary thresholding: adaptively choose method so strokes become white on black
    # Use Otsu for automatic thresholding
    _, binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray_blurred, cleaned

def detect_circle(gray_image: np.ndarray, binary_image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Detect the main circle (clock boundary). Prefer HoughCircles on blurred grayscale; fallback to contours."""
    h, w = gray_image.shape[:2]
    min_dim = min(h, w)

    # Try HoughCircles on blurred grayscale
    try:
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min_dim / 8),
            param1=100,   # Canny high threshold
            param2=30,    # accumulator threshold (smaller->more false circles)
            minRadius=int(min_dim * 0.12),
            maxRadius=int(min_dim * 0.48)
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # choose the circle with largest radius
            circles = sorted(circles, key=lambda c: c[2], reverse=True)
            cx, cy, r = circles[0]
            # Basic sanity check: circle inside image bounds
            if r > 10 and 0 < cx < w and 0 < cy < h:
                return (int(cx), int(cy), int(r))
    except Exception:
        # fall through to contour method
        pass

    # Fallback: use contours on the binary image (which has strokes white on black)
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 10:
            continue
        # circularity measure: how well contour fits an ideal circle
        circle_area = math.pi * (radius ** 2)
        if circle_area <= 0:
            continue
        circularity = area / circle_area  # 0..1 (1 is perfect)
        # prefer larger radius and higher circularity
        score = circularity * (radius / (min_dim / 2.0))
        if score > best_score and circularity > 0.35:
            best_score = score
            best = (int(x), int(y), int(radius))
    return best

def detect_numbers(orig_image: np.ndarray, binary_image: np.ndarray, ocr_reader, circle_info: Tuple) -> List[Dict]:
    """
    Detect digits 1-12 around the clock using EasyOCR.
    Uses the original RGB image to preserve text details, but crops to an annular
    region around the detected circle and applies adaptive preprocessing for OCR.
    Returns a list of detections with global coordinates.
    """
    numbers = []
    cx, cy, radius = circle_info
    h, w = binary_image.shape[:2]

    try:
        # Build annular mask (slightly outside and inside the radius)
        outer_r = int(radius * 1.15)
        inner_r = int(radius * 0.55)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), outer_r, 255, -1)
        cv2.circle(mask, (cx, cy), inner_r, 0, -1)

        # Work on the original RGB for OCR cropping
        if orig_image.ndim == 3 and orig_image.shape[2] == 4:
            rgb = orig_image[:, :, :3]
        elif orig_image.ndim == 3 and orig_image.shape[2] == 3:
            rgb = orig_image
        else:
            rgb = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2RGB)

        # Crop bounding box around the annulus to limit OCR area
        x_min = max(cx - outer_r, 0)
        x_max = min(cx + outer_r, w)
        y_min = max(cy - outer_r, 0)
        y_max = min(cy + outer_r, h)

        if x_max <= x_min or y_max <= y_min:
            return []

        crop_rgb = rgb[y_min:y_max, x_min:x_max]
        crop_mask = mask[y_min:y_max, x_min:x_max]

        # Convert to gray and apply mask so OCR focuses on digits
        crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        crop_masked = cv2.bitwise_and(crop_gray, crop_gray, mask=crop_mask)

        # Adaptive threshold/Otsu and invert if needed so digits are dark on white
        _, thresh = cv2.threshold(crop_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Determine if we need to invert (we want dark text on light background for OCR)
        # If most of the crop is white (background), keep; otherwise invert
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)

        # Resize up to help OCR
        scale = max(1.0, min(3.0, 800.0 / max(thresh.shape)))  # scale so the largest side ~800px, but <=3x
        new_w = int(thresh.shape[1] * scale)
        new_h = int(thresh.shape[0] * scale)
        thresh_resized = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert to 3-channel RGB for EasyOCR
        ocr_input = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2RGB)

        # Run OCR
        results = ocr_reader.readtext(ocr_input)

        for (bbox, text, confidence) in results:
            cleaned_text = ''.join(filter(str.isdigit, text))
            if not cleaned_text:
                continue
            try:
                val = int(cleaned_text)
            except Exception:
                continue
            if not (1 <= val <= 12):
                continue
            if confidence < 0.25:
                # skip very low confidence
                continue

            # bbox coordinates are relative to the resized crop; map back to original image coords
            # bbox is list of 4 points [(x1,y1),(x2,y2)...]
            bbox = np.array(bbox).astype(float)
            # scale back to crop coords
            bbox[:, 0] = bbox[:, 0] / scale
            bbox[:, 1] = bbox[:, 1] / scale
            # map to full image coords by adding offsets
            bbox[:, 0] += x_min
            bbox[:, 1] += y_min

            center_x = int(np.mean(bbox[:, 0]))
            center_y = int(np.mean(bbox[:, 1]))

            # ensure it's reasonably inside the annular zone
            dist = math.hypot(center_x - cx, center_y - cy)
            if dist < inner_r * 0.9 or dist > outer_r * 1.05:
                # not in expected ring; skip
                continue

            numbers.append({
                'text': str(val),
                'position': (center_x, center_y),
                'confidence': float(confidence),
                'bbox': bbox.tolist()
            })

    except Exception as e:
        st.warning(f"OCR detection encountered an issue: {e}")

    return numbers

def detect_hands(gray_image: np.ndarray, binary_image: np.ndarray, circle_info: Tuple) -> List[Dict]:
    """Detect line-like hands inside the clock face using Canny+Hough with masking."""
    hands = []
    cx, cy, radius = circle_info

    # Create mask that restricts to interior of clock
    mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(radius * 0.9), 255, -1)

    # Edge detection on blurred grayscale
    edges = cv2.Canny(gray_image, 50, 150)
    # Mask edges so only interior lines remain
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)

    # HoughLinesP to detect segments
    lines = cv2.HoughLinesP(
        edges_masked,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(radius * 0.15),
        maxLineGap=15
    )

    if lines is None:
        return []

    valid_hands = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # endpoints distances from center
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)

        # both endpoints should be inside circle-ish
        if d1 > radius * 1.05 or d2 > radius * 1.05:
            continue

        length = math.hypot(x2 - x1, y2 - y1)
        if length < radius * 0.12:
            continue

        # compute distance of line to center (perpendicular)
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        denom = math.hypot(A, B)
        if denom == 0:
            continue
        distance_to_center = abs(A * cx + B * cy + C) / denom

        # Accept lines that are reasonably close to center
        if distance_to_center > radius * 0.5:
            continue

        # choose the tip as the endpoint further from center
        if d1 > d2:
            tip_x, tip_y = x1, y1
        else:
            tip_x, tip_y = x2, y2

        # angle from 12 o'clock measured clockwise:
        vx = tip_x - cx
        vy = tip_y - cy  # positive down
        angle = math.atan2(vx, -vy)  # 0 is 12:00, increases clockwise
        angle = (angle + 2 * math.pi) % (2 * math.pi)

        valid_hands.append({
            'start': (x1, y1),
            'end': (x2, y2),
            'length': length,
            'angle': angle,
            'distance_to_center': distance_to_center,
            'tip': (tip_x, tip_y)
        })

    # sort by length (descending) and take top 2 (minute, hour)
    valid_hands.sort(key=lambda h: h['length'], reverse=True)
    return valid_hands[:2]

def calculate_time_from_hands(hands: List[Dict]) -> Optional[Tuple[int, int]]:
    """Given list of (preferably two) hands with 'angle' measured as above, return (hour, minute)."""
    if len(hands) < 2:
        return None

    # assume longer is minute, shorter is hour
    minute_hand = hands[0]
    hour_hand = hands[1]

    minute_angle = minute_hand['angle']
    hour_angle = hour_hand['angle']

    # convert angles to minute and hour positions
    minute_pos = int(round((minute_angle / (2 * math.pi)) * 60)) % 60

    # hour position: fractional hour index 0..11
    hour_pos_raw = (hour_angle / (2 * math.pi)) * 12.0
    # incorporate minute into hour estimate (approx)
    hour_fraction = hour_pos_raw % 12.0
    hour_int = int(hour_fraction)  # 0..11
    # convert to 1..12 display
    display_hour = hour_int if hour_int != 0 else 12
    # Map 0->12 for readability
    if display_hour == 0:
        display_hour = 12

    return display_hour, minute_pos

def score_clock_drawing(circle_info: Optional[Tuple], hands: List[Dict], 
                       numbers: List[Dict]) -> Dict:
    """Score the clock drawing and provide feedback with flexible time acceptance"""
    score = 0
    max_score = 10
    feedback = []
    detected_time = None

    # Clock face scoring (3 points)
    if circle_info:
        score += 3
        feedback.append("✓ Clock face detected correctly")
    else:
        feedback.append("✗ Clock face not detected clearly")
        return {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'feedback': feedback,
            'detected_time': None
        }

    # Numbers scoring (4 points)
    detected_numbers = len(numbers)

    if detected_numbers >= 8:
        score += 4
        feedback.append(f"✓ Excellent number detection ({detected_numbers} numbers found)")
    elif detected_numbers >= 4:
        score += 3
        feedback.append(f"◐ Good number detection ({detected_numbers} numbers found)")
    elif detected_numbers >= 2:
        score += 2
        feedback.append(f"◐ Some numbers detected ({detected_numbers} numbers found)")
    else:
        feedback.append(f"✗ Poor number detection ({detected_numbers} numbers found)")

    # Hands scoring (3 points)
    if len(hands) >= 2:
        score += 2
        feedback.append("✓ Both clock hands detected")

        time_result = calculate_time_from_hands(hands)
        if time_result:
            hour_pos, minute_pos = time_result
            detected_time = f"{hour_pos:02d}:{minute_pos:02d}"
            feedback.append(f"Detected time: {detected_time}")

            # Flexible acceptance around target 10:10
            hour_acceptable = hour_pos in [9, 10, 11]
            minute_acceptable = (8 <= minute_pos <= 15)  # 10 ± 5

            if hour_acceptable and minute_acceptable:
                score += 1
                feedback.append("✓ Hands pointing to approximately correct time range!")
            elif hour_acceptable or minute_acceptable:
                feedback.append("◐ One hand is in the correct range")
            else:
                feedback.append("◐ Hands detected but time is not in the target range")
        else:
            feedback.append("◐ Could not determine time from hand positions")
    elif len(hands) == 1:
        score += 1
        feedback.append("◐ Only one hand detected")
    else:
        feedback.append("✗ Clock hands not detected clearly")

    percentage = (score / max_score) * 100

    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage,
        'feedback': feedback,
        'detected_time': detected_time
    }

def create_analysis_visualization(original_image: np.ndarray, binary_image: np.ndarray,
                                circle_info: Optional[Tuple], hands: List[Dict],
                                numbers: List[Dict]) -> np.ndarray:
    """Create visualization showing detected components (returns RGB image)."""
    # Prepare a color version to draw on
    if original_image.ndim == 3 and original_image.shape[2] == 4:
        vis = original_image[:, :, :3].copy()
    elif original_image.ndim == 3 and original_image.shape[2] == 3:
        vis = original_image.copy()
    else:
        vis = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

    # ensure uint8
    vis = vis.astype(np.uint8)

    # draw circle
    if circle_info:
        cx, cy, radius = circle_info
        cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)

    # draw hands
    for i, hand in enumerate(hands):
        color = (255, 0, 0) if i == 0 else (0, 0, 255)
        cv2.line(vis, hand['start'], hand['end'], color, 3)
        tip = (int(hand['tip'][0]), int(hand['tip'][1]))
        cv2.circle(vis, tip, 4, color, -1)
        mid_x = int((hand['start'][0] + hand['end'][0]) / 2)
        mid_y = int((hand['start'][1] + hand['end'][1]) / 2)
        label = "M" if i == 0 else "H"
        cv2.putText(vis, label, (mid_x - 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # draw numbers
    for number in numbers:
        x, y = number['position']
        cv2.circle(vis, (x, y), 8, (255, 255, 0), -1)
        cv2.putText(vis, number['text'], (x - 10, y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return vis

def main():
    st.set_page_config(
        page_title="Clock Drawing Test",
        page_icon="🕒",
        layout="wide"
    )

    st.title("🕒 Clock Drawing Test (CDT)")
    st.markdown("### AI-Powered Analysis for Cognitive Assessment")

    # Initialize session state
    if 'circle_detected' not in st.session_state:
        st.session_state.circle_detected = False
    if 'circle_info' not in st.session_state:
        st.session_state.circle_info = None
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'canvas_hash' not in st.session_state:
        st.session_state.canvas_hash = None

    st.markdown("""
    **Instructions:** Please draw a clock showing **10:10** on the canvas below.

    - **Step 1:** Draw a circle for the clock face first and detect it
    - **Step 2:** Add numbers 1-12 around the clock
    - **Step 3:** Draw two hands pointing to 10:10 (hour hand between 10-11, minute hand at 2)

    **Note:** You must detect the circle first before proceeding with full analysis!
    """)

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Drawing Canvas")

        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"clock_canvas_{st.session_state.canvas_key}"
        )

        # Button layout
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

        # Real-time canvas change detection (light version)
        if canvas_result.image_data is not None and st.session_state.circle_detected:
            current_image = canvas_result.image_data.astype(np.uint8)
            if not np.any(current_image[:, :, 3] > 0):
                st.session_state.circle_detected = False
                st.session_state.circle_info = None
                st.session_state.canvas_hash = None
            elif hasattr(st.session_state, 'canvas_hash') and st.session_state.canvas_hash is not None:
                current_hash = hash(current_image.tobytes())
                if current_hash != st.session_state.canvas_hash:
                    gray_blurred, cleaned = preprocess_image(current_image)
                    if np.sum(cleaned) < 1000:  # Very little content
                        st.session_state.circle_detected = False
                        st.session_state.circle_info = None
                        st.session_state.canvas_hash = None

    with col2:
        st.subheader("Analysis Results")

        # Handle button click
        if detect_circle_button and canvas_result.image_data is not None:
            image_data = canvas_result.image_data.astype(np.uint8)

            # Check if canvas is actually empty by looking at alpha channel
            if not np.any(image_data[:, :, 3] > 0):
                st.session_state.circle_detected = False
                st.session_state.circle_info = None
                st.warning("Please draw something on the canvas first!")
                return

            # Store current canvas state hash to detect if canvas changes
            canvas_hash = hash(image_data.tobytes())

            if not st.session_state.circle_detected:
                # Perform circle detection
                with st.spinner("Detecting clock circle..."):
                    try:
                        gray_blurred, cleaned = preprocess_image(image_data)

                        if np.sum(cleaned) < 1000:
                            st.session_state.circle_detected = False
                            st.session_state.circle_info = None
                            st.error("❌ Not enough drawing content. Please draw a clearer circle.")
                            return

                        circle_info = detect_circle(gray_blurred, cleaned)

                        if circle_info:
                            st.session_state.circle_detected = True
                            st.session_state.circle_info = circle_info
                            st.session_state.canvas_hash = canvas_hash
                            cx, cy, radius = circle_info
                            st.success("✅ Circle detected successfully!")
                            st.info(f"Circle found at center ({cx}, {cy}) with radius {radius}")
                            st.info("You can now draw numbers and hands, then click 'Analyze Drawing' for full analysis.")

                            vis_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
                            cv2.circle(vis_image, (cx, cy), radius, (0, 255, 0), 2)
                            cv2.circle(vis_image, (cx, cy), 3, (0, 255, 0), -1)
                            st.image(vis_image, caption="Detected Circle (Green)", width=300)
                        else:
                            st.session_state.circle_detected = False
                            st.session_state.circle_info = None
                            st.error("❌ No circle detected. Please draw a clearer circle.")
                            st.info("Try drawing a more complete circular shape.")

                    except Exception as e:
                        st.error(f"Error during circle detection: {str(e)}")
            else:
                # Perform full analysis
                with st.spinner("Performing full analysis..."):
                    try:
                        # Check canvas hasn't drastically changed
                        if hasattr(st.session_state, 'canvas_hash') and canvas_hash != st.session_state.canvas_hash:
                            gray_blurred, cleaned = preprocess_image(image_data)
                            circle_still_exists = detect_circle(gray_blurred, cleaned)

                            if not circle_still_exists:
                                st.session_state.circle_detected = False
                                st.session_state.circle_info = None
                                st.error("❌ Circle no longer detected in current drawing. Please detect circle again.")
                                return
                            else:
                                st.session_state.circle_info = circle_still_exists
                                st.session_state.canvas_hash = canvas_hash

                        ocr_reader = load_ocr_reader()
                        gray_blurred, cleaned = preprocess_image(image_data)
                        circle_info = st.session_state.circle_info

                        if not circle_info:
                            st.error("❌ No valid circle found. Please detect circle first.")
                            return

                        numbers = detect_numbers(image_data, cleaned, ocr_reader, circle_info)
                        hands = detect_hands(gray_blurred, cleaned, circle_info)

                        scoring_result = score_clock_drawing(circle_info, hands, numbers)

                        st.success("✅ Full analysis complete!")

                        st.metric(
                            "Overall Score",
                            f"{scoring_result['score']}/{scoring_result['max_score']}",
                            f"{scoring_result['percentage']:.1f}%"
                        )

                        st.subheader("Detailed Feedback:")
                        for feedback_item in scoring_result['feedback']:
                            if feedback_item.startswith("✓"):
                                st.success(feedback_item)
                            elif feedback_item.startswith("◐"):
                                st.warning(feedback_item)
                            elif feedback_item.startswith("✗"):
                                st.error(feedback_item)
                            else:
                                st.info(feedback_item)

                        with st.expander("🔍 Detection Details"):
                            cx, cy, radius = circle_info
                            st.write(f"**Circle Detection:** ✓ Detected")
                            st.write(f"- Center: ({cx}, {cy})")
                            st.write(f"- Radius: {radius}")

                            st.write(f"**Numbers Detected:** {len(numbers)}")
                            if numbers:
                                for num in numbers:
                                    st.write(f"- Number '{num['text']}' at {num['position']} (conf: {num['confidence']:.2f})")

                            st.write(f"**Hands Detected:** {len(hands)}")
                            if hands:
                                for i, hand in enumerate(hands):
                                    hand_type = "Minute hand (longer)" if i == 0 else "Hour hand (shorter)"
                                    angle_deg = math.degrees(hand['angle'])
                                    st.write(f"- {hand_type}: length {hand['length']:.1f}px, angle {angle_deg:.1f}°")

                            if scoring_result['detected_time']:
                                st.write(f"**Detected Time:** {scoring_result['detected_time']}")

                        st.subheader("Visual Analysis")
                        vis_image = create_analysis_visualization(
                            image_data, cleaned, circle_info, hands, numbers
                        )
                        st.image(vis_image, caption="Analysis: Green=Circle, Red=Minute Hand, Blue=Hour Hand, Yellow=Numbers")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.error("Please try again or contact support if the issue persists.")

        # Status display
        if st.session_state.circle_detected:
            st.info("✅ Circle detected - You can now perform full analysis by clicking 'Analyze Drawing'!")
        else:
            st.info("⏳ Please draw a circle and click 'Detect Circle' to begin analysis.")

    # Additional information
    st.markdown("---")
    st.markdown("""
    ### About the Clock Drawing Test

    The Clock Drawing Test (CDT) is a widely used screening tool for cognitive impairment and dementia. 
    This AI-powered version analyzes your drawing in two steps:

    **Step 1: Circle Detection**
    - Detects the clock face boundary
    - Must be completed before proceeding

    **Step 2: Full Analysis**
    - **Numbers**: Placement of digits 1-12 around the clock (only within the detected circle)
    - **Clock Hands**: Detection of hour and minute hands (only inside the circle)
    - **Time Accuracy**: Flexible scoring for times around 10:10

    **Scoring Criteria:**
    - **Perfect 10:10**: Hour hand pointing between 9-11, minute hand at 8-15 minutes
    - **Circle**: 3 points for clear clock face
    - **Numbers**: Up to 4 points based on quantity detected
    - **Hands**: Up to 3 points for detection and time accuracy

    **Note**: This is a demonstration tool and should not be used for medical diagnosis. 
    Please consult healthcare professionals for proper cognitive assessment.
    """)

if __name__ == "__main__":
    main()
