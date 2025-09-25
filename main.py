import streamlit as st
from streamlit_drawable_canvas import st_canvas
import math
import time
import pandas as pd
import os
import random
import numpy as np

# ---------- Settings ----------
TOTAL_TASKS = 6
CANVAS_SIZE = 400

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

def evaluate_clock(user_numbers, hour_angle, minute_angle):
    correct_numbers = [str(i) for i in range(1,13)]
    missing_numbers = [n for n in correct_numbers if n not in user_numbers]
    misplaced_numbers = [n for n in user_numbers if n not in correct_numbers]
    
    # Evaluate hands (simplified)
    correct_hour_angle = math.radians((10-3)*30 + (10/60)*30)  # example 10:10
    correct_minute_angle = math.radians((10-15)*6)
    hand_score = 2 - int(abs(hour_angle-correct_hour_angle) > 0.5) - int(abs(minute_angle-correct_minute_angle) > 0.5)
    hand_score = max(hand_score,0)
    
    number_score = max(0, 12 - len(missing_numbers) - len(misplaced_numbers))
    total_score = number_score + hand_score
    final_score = (total_score/14)*100
    return final_score, missing_numbers, misplaced_numbers, hand_score

# ---------- Initialize session ----------
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
    key="canvas"
)

# ---------- Submit Drawing ----------
if st.button("Submit Drawing"):
    time_taken = time.time() - st.session_state.start_time
    task_type = st.session_state.current_task

    if task_type == "Clock":
        # Simulate automatic detection
        # In real case, extract numbers & angles from drawing
        user_numbers = [str(i) for i in range(1,13)]  # assume user drew all correctly
        hour_angle = math.radians((10-3)*30 + (10/60)*30)
        minute_angle = math.radians((10-15)*6)
        final_score, missing_numbers, misplaced_numbers, hand_score = evaluate_clock(user_numbers, hour_angle, minute_angle)
    else:
        # Approximate shape score
        final_score = random.randint(60,100)
        missing_numbers = []
        misplaced_numbers = []
        hand_score = 0

    st.session_state.all_scores.append(final_score)
    df_result = pd.DataFrame([[task_type, round(final_score,2), round(time_taken,2), missing_numbers, misplaced_numbers, hand_score]],
                             columns=["Task_Type","Score","Time_seconds","Missing_Numbers","Misplaced_Numbers","Hand_Score"])
    df_result.to_csv(results_file, mode='a', header=False, index=False)

    st.session_state.task_count += 1
    st.session_state.current_task = None
    st.success(f"Task submitted! Task {st.session_state.task_count+1 if st.session_state.task_count<TOTAL_TASKS else 'finished'} is ready.")
