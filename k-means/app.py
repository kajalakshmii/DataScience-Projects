import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\\internship\\project in regression\\k-means\\KNN_Dataset.csv')

st.title("Diabetes Prediction with KNN")

page = st.sidebar.selectbox("Select a page", ["Dataset Overview", "Missing Values", "Data Visualization", "Model Evaluation"])

X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

if page == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Number of rows:", len(dataset))
    st.dataframe(dataset.head())
    st.dataframe(dataset.info())
    st.dataframe(dataset.describe())

elif page == "Missing Values":
    st.subheader("Missing Values")
    st.dataframe(dataset.isnull().sum())

elif page == "Data Visualization":
    st.subheader("Scatter Plot of Age and BMI with Outcome Differentiation")
    plt.figure(figsize=[10, 6], dpi=80)
    plt.scatter(dataset[dataset['Outcome'] == 0]['BMI'], dataset[dataset['Outcome'] == 0]['Age'], color="blue", label="Outcome 0")
    plt.scatter(dataset[dataset['Outcome'] == 1]['BMI'], dataset[dataset['Outcome'] == 1]['Age'], color="red", label="Outcome 1")
    plt.xlabel('BMI')
    plt.ylabel('Age')
    plt.title("Scatter Plot of Age and BMI in Diabetes Data with Outcome Differentiation", weight='bold', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.legend()
    st.pyplot(plt)

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in zero_not_accepted:
        dataset[column] = dataset[column].replace(0, np.NaN)
        mean = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.NaN, mean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
