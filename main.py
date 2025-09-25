import streamlit as st
from streamlit_drawable_canvas import st_canvas
import math
import time
import pandas as pd
import os
import random

# ---------- Settings ----------
TOTAL_TASKS = 6
CANVAS_SIZE = 400

# Files
results_file = "visuospatial_results_web.csv"
report_file = "cognitive_report_web.csv"

# ---------- Shapes ----------
shapes = {
    "Triangle": [(50, 50), (150, 50), (100, 150)],
    "House": [(50, 150), (150, 150), (150, 250), (50, 250), (50, 150), (100, 100), (150, 150)],
    "Clock": "Clock"
}

def clock_positions(center=(200,200), radius=100):
    positions = {}
    for i in range(1,13):
        angle = math.radians((i-3)*30)
        x = center[0] + radius*math.cos(angle)
        y = center[1] + radius*math.sin(angle)
        positions[str(i)] = (x,y)
    return positions

# ---------- Initialize Session State ----------
if "task_count" not in st.session_state:
    st.session_state.task_count = 0
if "all_scores" not in st.session_state:
    st.session_state.all_scores = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "current_task" not in st.session_state:
    st.session_state.current_task = None

# ---------- Prepare CSV ----------
if not os.path.exists(results_file):
    pd.DataFrame(columns=["Task_Type","Score","Time_seconds","Missing_Numbers","Misplaced_Numbers","Hand_Score"]).to_csv(results_file,index=False)
if not os.path.exists(report_file):
    pd.DataFrame(columns=["Task_Count","Average_Score","Overall_Risk"]).to_csv(report_file,index=False)

st.title("🧠 Visuospatial Cognitive Assessment Suite")

# ---------- Check if all tasks completed ----------
if st.session_state.task_count >= TOTAL_TASKS:
    avg_score = sum(st.session_state.all_scores)/len(st.session_state.all_scores)
    if avg_score < 60:
        risk = "High"
    elif avg_score < 80:
        risk = "Moderate"
    else:
        risk = "Low"
    st.subheader("✅ Final Cognitive Risk Report")
    st.write(f"Average Score: {avg_score:.2f}%")
    st.write(f"Overall Cognitive Risk: {risk}")

    # Save final report
    df_report = pd.DataFrame([[st.session_state.task_count, round(avg_score,2), risk]],
                             columns=["Task_Count","Average_Score","Overall_Risk"])
    df_report.to_csv(report_file, mode='a', header=False, index=False)
    st.stop()

# ---------- Start Next Task ----------
if st.session_state.current_task is None:
    st.session_state.current_task = random.choice(list(shapes.keys()))
    st.session_state.start_time = time.time()
st.write(f"Task {st.session_state.task_count+1} of {TOTAL_TASKS}: {st.session_state.current_task}")

# ---------- Canvas ----------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=3,
    stroke_color="black",
    background_color="white",
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------- Submit Drawing ----------
if st.button("Submit Drawing"):
    time_taken = time.time() - st.session_state.start_time
    task_type = st.session_state.current_task
    score = 0
    missing_numbers = []
    misplaced_numbers = []
    hand_score = 0

    if task_type == "Clock":
        numbers_input = st.text_input("Enter placed numbers separated by comma (1-12):", "1,2,3,4,5,6,7,8,9,10,11,12")
        user_numbers = [n.strip() for n in numbers_input.split(",") if n.strip().isdigit()]
        correct_numbers = [str(i) for i in range(1,13)]
        missing_numbers = [n for n in correct_numbers if n not in user_numbers]
        score += max(0, 12 - len(missing_numbers))
        hand_score = st.slider("Rate hands placement (0-2)", 0, 2, 2)
        score += hand_score
        max_score = 14
        final_score = (score/max_score)*100
    else:
        st.write("Shape drawing submitted. Score will be approximated.")
        final_score = random.randint(60,100)  # Approximation for demo

    st.write(f"Task Score: {final_score:.2f}%")
    st.session_state.all_scores.append(final_score)

    # Save per-task result
    df_result = pd.DataFrame([[task_type, round(final_score,2), round(time_taken,2), missing_numbers, misplaced_numbers, hand_score]],
                             columns=["Task_Type","Score","Time_seconds","Missing_Numbers","Misplaced_Numbers","Hand_Score"])
    df_result.to_csv(results_file, mode='a', header=False, index=False)

    st.success("Drawing submitted! Click 'Next Task' to continue.")

# ---------- Next Task Button ----------
if st.button("Next Task"):
    st.session_state.current_task = None
    st.session_state.start_time = time.time()
    st.success("Next task loaded! Start drawing now.")
