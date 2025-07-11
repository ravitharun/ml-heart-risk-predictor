# app.py
import streamlit as st
import pandas as pd                      # For loading dataset
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LogisticRegression

Data=pd.read_csv(r'C:\Users\Dell\Downloads\AIML\PROJECTS\project1\heart.csv')
st.title("❤️ Heart Disease Risk Predictor")

with st.expander("ℹ️ About the Project"):
    st.write(
        "This project uses a Logistic Regression model to predict the likelihood of Heart Disease "
        "based on key medical features such as **age**, **sex**, **chest pain type (cp)**, "
        "**resting blood pressure (trestbps)**, **cholesterol**, **maximum heart rate (thalach)**, "
        "**ST depression (oldpeak)**, and more."
    )
    st.write(
        "The model analyzes these features and provides a simple output: **Yes (1)** if the person "
        "is likely to have heart disease, or **No (0)** if not."
    )
    st.write("This tool is designed for educational and exploratory purposes.")
age = st.slider("Age", 20, 80, help="Patient's age in years")
sex = st.slider("Sex (0 = female, 1 = male)", 0, 1, help="Select 0 for female, 1 for male")
cp = st.slider("Chest Pain Type (0–3)", 0, 3, help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, help="Typical resting blood pressure")
chol = st.slider("Serum Cholesterol (mg/dL)", 100, 400, help="Serum cholesterol in mg/dL")
fbs = st.slider("Fasting Blood Sugar > 120 mg/dL (1 = true, 0 = false)", 0, 1)
restecg = st.slider("Resting ECG Results (0–2)", 0, 2, help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
exang = st.slider("Exercise-Induced Angina (1 = yes, 0 = no)", 0, 1)
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1, help="ST depression induced by exercise")
slope = st.slider("Slope of the ST segment (0–2)", 0, 2, help="0: Upsloping, 1: Flat, 2: Downsloping")
ca = st.slider("Number of Major Vessels Colored (0–4)", 0, 4)
thal = st.slider("Thalassemia Type (1 = normal, 2 = fixed defect, 3 = reversible defect)", 1, 3)
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)


X = Data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
          'restecg', 'thalach', 'exang', 'oldpeak', 
          'slope', 'ca', 'thal']]     # shape: (1025, 13)

y = Data['target']                    # shape: (1025,)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
predict = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach, exang, oldpeak, slope, ca, thal]])

if st.button("check"):
    if predict[0]==0:
        test_acc = model.score(x_test, y_test)
        st.write("Test Accuracy:", test_acc)
        st.title(f' (No) You dont have Heart Disease  ' )
    else:
        test_acc = model.score(x_test, y_test)
        st.write("Test Accuracy:", test_acc)
        st.title(f'(Yes) You  have Heart Disease   ' )



tab1, tab2 = st.tabs(["📈 Charts", "📋 Raw Data"])

with tab1:
    st.line_chart(Data['age'])

with tab2:
    st.dataframe(Data)
