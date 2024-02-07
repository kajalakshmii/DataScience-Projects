import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

diabetes_dataset = pd.read_csv("D:\\internship\\project in regression\\svm\\diabetes.csv")

scaler = StandardScaler()
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler.fit(X)

Y = diabetes_dataset['Outcome']
X = scaler.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

st.sidebar.title("Diabetes Prediction App")
selected_page = st.sidebar.selectbox("Select a page", ["Data Overview", "Model Evaluation", "Prediction"])

if selected_page == "Data Overview":
    st.write("Data Overview")
    st.write(diabetes_dataset.head())
    st.write("Shape of dataset:", diabetes_dataset.shape)
    st.write("Statistical measures of the dataset:")
    st.write(diabetes_dataset.describe())
    st.write("Number of diabetic and non-diabetic individuals:")
    st.write(diabetes_dataset['Outcome'].value_counts())
 
elif selected_page == "Model Evaluation":
    st.write("Model Evaluation")
    x_axis = st.selectbox("Select X-axis", diabetes_dataset.columns[:-1])
    y_axis = st.selectbox("Select Y-axis", diabetes_dataset.columns[:-1])

    fig, ax = plt.subplots(figsize=[25, 10], dpi=150)
    ax.scatter(diabetes_dataset[x_axis], diabetes_dataset[y_axis], color="red", label="Data points")
    slope, intercept = np.polyfit(diabetes_dataset[x_axis], diabetes_dataset[y_axis], 1)
    line = slope * diabetes_dataset[x_axis] + intercept
    ax.plot(diabetes_dataset[x_axis], line, color="blue", label="Regression line")
    ax.set_title(f"The relationship between {y_axis} with {x_axis}", weight='bold', fontsize=30)
    ax.set_xlabel(x_axis, fontsize=30)
    ax.set_ylabel(y_axis, fontsize=30)
    ax.grid()
    ax.legend()
    st.pyplot(fig)
            
                            
    train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(train_pred, Y_train)
    st.write("Accuracy score on train data:", train_accuracy)
    test_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(test_pred, Y_test)
    st.write("Accuracy score on test data:", test_accuracy)

elif selected_page == "Prediction":
    st.write("Prediction")
    input_data = st.text_input("Enter input data separated by commas (e.g., 7,196,90,0,0,39.8,0.451,41):")
    if input_data:
        data_changed = np.asarray(tuple(map(float, input_data.split(','))))
        data_reshaped = data_changed.reshape(1, -1)
        std_data = scaler.transform(data_reshaped)
        prediction = classifier.predict(std_data)
        if prediction == 1:
            st.write('The person is Diabetic')
        else:
            st.write('The person is Non-Diabetic')
    else:
        st.write("Please enter input data")

