# import streamlit as st
# import pandas as pd
# import joblib

# # Load model and encoders
# model = joblib.load("best_model.pkl")
# encoders = joblib.load("encoders.pkl")

# st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼")

# st.title("💼 Employee Salary Prediction")
# st.write("Enter employee details to predict salary category.")

# # ---- Input Fields ----
# age = st.slider("Age", 17, 75, 30)
# workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
# educational_num = st.slider("Educational Number", 5, 16, 10)
# marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
# occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
# relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
# race = st.selectbox("Race", encoders['race'].classes_)
# gender = st.selectbox("Gender", encoders['gender'].classes_)

# # ---- Convert inputs to DataFrame ----
# input_df = pd.DataFrame([{
#     'age': age,
#     'workclass': encoders['workclass'].transform([workclass])[0],
#     'educational-num': educational_num,
#     'marital-status': encoders['marital-status'].transform([marital_status])[0],
#     'occupation': encoders['occupation'].transform([occupation])[0],
#     'relationship': encoders['relationship'].transform([relationship])[0],
#     'race': encoders['race'].transform([race])[0],
#     'gender': encoders['gender'].transform([gender])[0],
#     # Fill other columns with default values if model trained with more features
#     'capital-gain': 0,
#     'hours-per-week': 40,
#     'native-country': 0
# }])

# # ---- Predict Button ----
# if st.button("Predict Salary"):
#     prediction = model.predict(input_df)[0]
#     if prediction == 0:
#         st.success("💰 Predicted Salary: <= 50K")
#     else:
#         st.success("💰 Predicted Salary: > 50K")




import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ Load Model & Encoders
# --------------------------
best_model = joblib.load("best_salary_model.pkl")
encoders = joblib.load("salary_encoders.pkl")
data = pd.read_csv("Salary Data.csv")

# --------------------------
# 2️⃣ Dashboard Header
# --------------------------
st.title("💼 Employee Salary Prediction Dashboard")
st.write("This app predicts employee salaries based on age, gender, education, job title, and years of experience.")
st.markdown("---")

# --------------------------
# 3️⃣ User Input Section
# --------------------------
st.header("📌 Enter Employee Details")

# Fill missing values in dataset before getting unique dropdown values
data = data.fillna({
    "Gender": data["Gender"].mode()[0],
    "Education Level": data["Education Level"].mode()[0],
    "Job Title": data["Job Title"].mode()[0],
    "Age": data["Age"].median(),
    "Years of Experience": data["Years of Experience"].median(),
    "Salary": data["Salary"].median()
})

age = st.slider("Age", 18, 60, 20)
gender = st.selectbox("Gender", data["Gender"].unique())
education = st.selectbox("Education Level", data["Education Level"].unique())
job_title = st.selectbox("Job Title", data["Job Title"].unique())
experience = st.slider("Years of Experience", 0, 40, 5)

if st.button("🔍 Predict Salary"):
    # Encode categorical inputs
    gender_encoded = encoders["Gender"].transform([gender])[0]
    edu_encoded = encoders["Education Level"].transform([education])[0]
    job_encoded = encoders["Job Title"].transform([job_title])[0]

    # Create input DataFrame
    input_df = pd.DataFrame([[age, gender_encoded, edu_encoded, job_encoded, experience]],
                            columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

    # Predict
    prediction = best_model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: **${prediction:,.2f}**")

st.markdown("---")

# --------------------------
# 4️⃣ Model Performance Section
# --------------------------
st.header("📊 Model Performance Insights")

# Example R2 scores (replace with actual results if saved separately)
model_scores = {
    "LinearRegression": 0.85,
    "RandomForestRegressor": 0.92,
    "KNeighborsRegressor": 0.81,
    "SVR": 0.78,
    "GradientBoostingRegressor": 0.90
}

fig, ax = plt.subplots()
ax.bar(model_scores.keys(), model_scores.values(), color='skyblue')
ax.set_ylabel("R² Score")
ax.set_title("Model Comparison")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("---")

# --------------------------
# 5️⃣ Data Insights Section
# --------------------------
st.header("📈 Salary Data Insights")

# Top 5 highest paid job titles
top_jobs = data.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(5)
st.subheader("💼 Top 5 Highest Paid Job Titles")
st.bar_chart(top_jobs)

# Average Salary by Education Level
edu_salary = data.groupby("Education Level")["Salary"].mean()
st.subheader("🎓 Average Salary by Education Level")
st.bar_chart(edu_salary)

# Salary vs Experience
st.subheader("📊 Salary vs Years of Experience")
st.line_chart(data.groupby("Years of Experience")["Salary"].mean())

# Gender-wise Average Salary
gender_salary = data.groupby("Gender")["Salary"].mean()
st.subheader("👨‍💼 Gender-wise Average Salary")
st.bar_chart(gender_salary)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit** & **Scikit-learn** | [LinkedIn](https://www.linkedin.com/in/sowjanya-kumar-gosala/)")
