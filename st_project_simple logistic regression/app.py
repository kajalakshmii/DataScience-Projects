import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("D:\\internship\\project in regression\\st_project_simple logistic regression\\diabetes.csv")

st.sidebar.title("Diabetes Prediction App")

selected_page = st.sidebar.selectbox("Select a page", ["Data Overview", "Scatter Plot", "Model Evaluation"])

if selected_page == "Data Overview":
    st.title("Data Overview")
    st.dataframe(df.head())

elif selected_page == "Scatter Plot":
    st.title("Scatter Plot")

    x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)

    fig, ax = plt.subplots(figsize=[25, 10], dpi=150)
    ax.scatter(df[x_axis], df[y_axis], color="red", label="Data points")
    slope, intercept = np.polyfit(df[x_axis], df[y_axis], 1)
    line = slope * df[x_axis] + intercept
    ax.plot(df[x_axis], line, color="blue", label="Regression line")
    ax.set_title(f"The relationship between {y_axis} with {x_axis}", weight='bold', fontsize=30)
    ax.set_xlabel(x_axis, fontsize=30)
    ax.set_ylabel(y_axis, fontsize=30)
    ax.grid()
    ax.legend()
    st.pyplot(fig)
elif selected_page == "Model Evaluation":
    st.title("Model Evaluation")

    X = df.drop(["Outcome"], axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)

    st.subheader("Model Evaluation:")
    st.write(f"Accuracy: {logReg.score(X_test, y_test)}")

    cm = confusion_matrix(y, logReg.predict(X))
    st.subheader("Confusion Matrix:")
    st.write(cm)

    st.subheader("Classification Report:")
    st.write(classification_report(y, logReg.predict(X)))
