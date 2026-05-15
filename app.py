

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ----------------------------
# Streamlit Page Config
# ----------------------------

st.set_page_config(page_title="AI Resume Screening System")

st.title("AI-Based Resume Screening and Candidate Shortlisting System")

st.write("Predict whether a candidate should be shortlisted.")

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv(
    r"C:\Users\fardeen\Downloads\resume dataset\ai_resume_screening.csv"
)

# ----------------------------
# Encode Target Column
# ----------------------------

label_encoder = LabelEncoder()

df["shortlisted"] = label_encoder.fit_transform(df["shortlisted"])

# Yes -> 1
# No -> 0

# ----------------------------
# Encode Education Level
# ----------------------------

education_encoder = LabelEncoder()

df["education_level"] = education_encoder.fit_transform(
    df["education_level"]
)

# ----------------------------
# Features and Target
# ----------------------------

X = df.drop("shortlisted", axis=1)

y = df["shortlisted"]

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Model Accuracy
# ----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

# ----------------------------
# User Input Section
# ----------------------------

st.header("Enter Candidate Details")

years_experience = st.slider(
    "Years of Experience",
    0,
    20,
    2
)

skills_match_score = st.slider(
    "Skills Match Score",
    0,
    100,
    50
)

education_level = st.selectbox(
    "Education Level",
    ["Bachelors", "Masters", "PhD"]
)

project_count = st.slider(
    "Project Count",
    0,
    20,
    2
)

resume_length = st.slider(
    "Resume Length",
    1,
    10,
    5
)

github_activity = st.slider(
    "GitHub Activity",
    0,
    100,
    50
)

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict"):

    # Encode education level input
    education_level_encoded = education_encoder.transform(
        [education_level]
    )[0]

    # Create input dataframe
    input_data = pd.DataFrame({
        "years_experience": [years_experience],
        "skills_match_score": [skills_match_score],
        "education_level": [education_level_encoded],
        "project_count": [project_count],
        "resume_length": [resume_length],
        "github_activity": [github_activity]
    })

    # Predict
    prediction = model.predict(input_data)

    # Convert prediction back
    result = label_encoder.inverse_transform(prediction)

    # Show result
    if result[0] == "Yes":
        st.success("Candidate is SHORTLISTED")
    else:
        st.error("Candidate is NOT SHORTLISTED")
