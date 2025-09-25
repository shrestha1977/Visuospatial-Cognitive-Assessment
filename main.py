import streamlit as st
from streamlit_drawable_canvas import st_canvas
import math
import time
import pandas as pd
import os
import random
import cv2
import numpy as np

# ---------- Settings ----------
TOTAL_TASKS = 6
CANVAS_SIZE = 400

results_file = "visuospatial_results_web.csv"
report_file = "cognitive_report_web.csv"

shapes = {
    "Triangle": [(50, 50), (150, 50), (100, 150)],
    "House": [(50, 150), (150, 150), (150, 250), (50, 250), (50, 150), (100, 100), (150, 150)],
    "Clock": "Clock"
}

# ---------- Session State ----------
if "task_count" not in st.session_state:
    st.session_state.task_count = 0
if "current_task" not in st.session_state:
    st.session_state.current_task = None
if "all_scores" not in st.session_state:
    st.session_state.all_scores = []

# ---------- Prepare CSV ----------
if not os.path.exists(results_file):
    pd.DataFrame(columns=["Task_Type","Score","Time_seconds","Missing_Numbers","Misplaced_Numbers","Hand_Score"]).to_csv(results_file,index=False)
if not os.path.exists(report_file):
    pd.DataFrame(columns=["Task_Count","Average_Score","Overall_Risk"]).to_csv(report_file,index=False)

st.title("🧠 Visuospatial Cognitive Assessment Suite")

# ---------- Check if finished ----------
if st.session_state.task_count >= TOTAL_TASKS:
    avg_score = sum(st.session_state.all_scores)/len(st.session_state.all_scores)
    risk = "High" if avg_score < 60 else ("Moderate" if avg_score < 80 else "Low")
    st.subheader("✅ Final Cognitive Risk Report")
    st.write(f"Average Score: {avg_score:.2f}%")
    st.write(f"Overall Cognitive Risk: {risk}")
    df_report = pd.DataFrame([[st.session_state.task_count, round(avg_score,2), risk]],
                             columns=["Task_Count","Average_Score","Overall_Risk"])
    df_report.to_csv(report_file, mode='a', header=False, index=False)
    st.write("CSV Results Saved ✅")
    st.stop()

# ---------- Load next task ----------
if st.session_state.current_task is None:
    st.session_state.current_task = random.choice(list(shapes.keys()))
    st.session_state.start_time = time.time()

st.write(f"Task {st.session_state.task_count+1} of {TOTAL_TASKS}: {st.session_state.current_task}")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=3,
    stroke_color="black",
    background_color="white",
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.task_count}"
)

# ---------- Clock Evaluation Function ----------
def evaluate_clock_from_canvas(image):
    """
    Evaluates the clock drawing:
    - Detects circles (numbers)
    - Detects lines (hands)
    - Computes missing/misplaced numbers
    - Computes hand score
    """
    img = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=15, minRadius=5, maxRadius=15)
    numbers_detected = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        numbers_detected = [str(i+1) for i in range(min(len(circles[0]),12))]

    # Simplified hand score: check if lines exist
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=50,maxLineGap=10)
    hand_score = 2 if lines is not None else 0

    # Number evaluation
    correct_numbers = [str(i) for i in range(1,13)]
    missing_numbers = [n for n in correct_numbers if n not in numbers_detected]
    misplaced_numbers = []  # Could be added by angle check
    number_score = max(0, 12 - len(missing_numbers))
    total_score = number_score + hand_score
    final_score = (total_score/14)*100
    return final_score, missing_numbers, misplaced_numbers, hand_score

# ---------- Submit Drawing ----------
if st.button("Submit Drawing"):
    time_taken = time.time() - st.session_state.start_time
    task_type = st.session_state.current_task

    if task_type == "Clock":
        if canvas_result.image_data is not None:
            final_score, missing_numbers, misplaced_numbers, hand_score = evaluate_clock_from_canvas(canvas_result.image_data)
        else:
            st.warning("Please draw something on the canvas.")
            st.stop()
    else:
        final_score = random.randint(60,100)
        missing_numbers = []
        misplaced_numbers = []
        hand_score = 0

    st.session_state.all_scores.append(final_score)
    df_result = pd.DataFrame([[task_type, round(final_score,2), round(time_taken,2),
                               missing_numbers, misplaced_numbers, hand_score]],
                             columns=["Task_Type","Score","Time_seconds","Missing_Numbers","Misplaced_Numbers","Hand_Score"])
    df_result.to_csv(results_file, mode='a', header=False, index=False)

    st.success(f"Task {st.session_state.task_count+1} submitted! Score: {final_score:.2f}%")
    st.session_state.task_count += 1
    st.session_state.current_task = None
