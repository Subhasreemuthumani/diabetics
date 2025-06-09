import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load dataset from local CSV and train model
# ---------------------------
@st.cache_data
def load_and_train():
    # Load dataset from local CSV file
    df = pd.read_csv('diabetes.csv')
    
    # Check if 'Outcome' column exists; if not, rename accordingly or notify
    if 'Outcome' not in df.columns:
        st.error("CSV file must contain 'Outcome' column as target variable.")
        return None, None
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf, df

rf, df = load_and_train()

if rf is None:
    st.stop()  # Stop app if dataset loading failed

# ---------------------------
# Feature Importance Plot
# ---------------------------
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Feature Importance from Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt.gcf())

# ---------------------------
# Streamlit App Interface
# ---------------------------

st.title("Diabetes Prediction App (Local CSV)")

st.write("""
This app uses a Random Forest model trained on the Pima Indians Diabetes dataset loaded from a local CSV file.
""")

# Input widgets for user
pregnancies = st.number_input('Pregnancies', 0, 20, 1)
glucose = st.number_input('Glucose Level', 0, 200, 120)
blood_pressure = st.number_input('Blood Pressure', 0, 140, 70)
skin_thickness = st.number_input('Skin Thickness', 0, 100, 20)
insulin = st.number_input('Insulin Level', 0, 900, 79)
bmi = st.number_input('BMI', 0.0, 70.0, 25.0)
dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.47)
age = st.number_input('Age', 10, 100, 33)

input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

if st.button('Predict'):
    prediction = rf.predict(input_data)
    if prediction[0] == 1:
        st.error("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No Diabetes")

st.write("---")

st.subheader("Feature Importance")
plot_feature_importance(rf, df.columns[:-1])
