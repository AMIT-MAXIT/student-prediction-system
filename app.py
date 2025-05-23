# 1. Imports
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("student_performance_data.csv")

# Features
X_data = df[["Assignment", "Exam", "Attendance", "Project", "Study Hours/Day"]].values
grades = df["Grade"].values

# Convert grades to numeric
grade_to_numeric = {'A': 6.0, 'B': 5.0, 'C': 4.0, 'D': 3.0, 'E': 2.0, 'F': 1.0}
numeric_to_grade = {v: k for k, v in grade_to_numeric.items()}
y_data = np.array([grade_to_numeric[g] for g in grades])

# Train model
model = LinearRegression()
model.fit(X_data, y_data)

# Streamlit page config
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# 🌟 Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #f7f9fc;
    }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        min-height: 100vh;
    }

    .title-underline {
        text-align: center;
        color: #1f2937;
        font-size: 3rem;
        padding-bottom: 10px;
        border-bottom: 4px solid #6366f1;
        display: inline-block;
        margin-bottom: 2rem;
        animation: fadeIn 2s ease;
    }

    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }

    

    .input-container:hover {
        box-shadow: 0px 12px 40px rgba(0, 0, 0, 0.15);
        transform: scale(1.01);
    }

    input, .stSlider > div {
        border-radius: 10px !important;
        border: 1px solid #6366f1 !important;
    }

    button[kind="primary"] {
        background-color: #6366f1;
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
    }

    button[kind="primary"]:hover {
        background-color: #4f46e5;
        transform: scale(1.05);
    }

    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #6366f1;
    }

    .prediction-result {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.5rem;
        color: #10b981;
    }

    .future-message {
        font-size: 1rem;
        background: #e0f7fa;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }

    </style>
    """, unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title-underline">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)

# Input container
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.header("📥 Enter Student Details")

# Input fields
assignment = st.number_input("📘 Assignment Marks", min_value=0, max_value=100, step=1)
exam = st.number_input("📝 Exam Marks", min_value=0, max_value=100, step=1)
attendance = st.number_input("📅 Attendance (%)", min_value=0, max_value=100, step=1)
project = st.number_input("🧪 Project Score", min_value=0, max_value=100, step=1)
study_hours_per_day = st.slider("📚 Study Hours per Day", min_value=0, max_value=5, value=2)

# Note
st.markdown(
    """
    <p style="color: #374151; font-size: 0.9rem;">
    <strong>Tip:</strong> Please provide realistic inputs to get accurate predictions.
    </p>
    """, unsafe_allow_html=True
)

# Predict button
if st.button("Predict"):
    input_data = np.array([[assignment, exam, attendance, project, study_hours_per_day]])
    predicted_numeric = model.predict(input_data)[0]
    rounded = round(predicted_numeric)
    rounded = max(1, min(6, rounded))
    grade = numeric_to_grade[rounded]

    st.markdown(f"<p class='prediction-result'>Predicted Grade: <strong>{grade}</strong></p>", unsafe_allow_html=True)

    # Future prediction
    future_map = {
        'A': "🌟  OUTSTANDING, You're destined to be a top scholar and leader!",
        'B': "📘 EXCELLENT, Strong performance! Keep sharpening your skills.",
        'C': "🛠️  GOOD, With more focus, you can climb higher!",
        'D': "🚀 AVERAGE, steadily and don't give up.",
        'E': "🧩 POOR, improvement needed, but success is possible!",
        'F': "⚡ FAIL, Urgent action needed. Seek help and work hard."
    }

    st.markdown(f"<div class='future-message'>{future_map[grade]}</div>", unsafe_allow_html=True)

    # Improvement tips
    improvement_map = {
        'A': "✅ Continue challenging yourself with complex projects and leadership roles.",
        'B': "📖 Revise difficult topics and practice sample papers.",
        'C': "🧠 Spend more time understanding basics and practice consistently.",
        'D': "🕒 Increase daily study time and stay disciplined with schedules.",
        'E': "🎯 Attend extra classes, seek tutoring, and rebuild weak areas.",
        'F': "🆘 Take strong action now: daily study routine, mentoring, and consistent hard work are crucial!"
    }

    st.warning(improvement_map[grade])

st.markdown('</div>', unsafe_allow_html=True)


# Adding a few more comments to further clarify parts of the code
# This is the model's input handling
# Each student input is represented as a feature (input) for prediction:
# - Assignment Marks: Numeric score representing performance in assignments.
# - Exam Marks: Numeric score from the final exam.
# - Attendance: Percentage of the classes attended.
# - Project Score: The score achieved on the project.
# - Study Hours per Day: The number of hours the student typically spends studying each day.

# The model is trained using the Linear Regression algorithm, where the input features 
# are used to predict the grade in numeric format (converted back to A, B, C, or D).

# In the case of predicting a grade, the model checks the inputs and computes the expected 
# grade, rounding it to the closest valid grade based on the numeric prediction.

# All information is displayed clearly for users to understand their current standing and potential future.
