import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = pd.read_csv("D:\\internship\\project in regression\\st_project_multiple logistic regression\\Iris.csv")

page_options = ["Data Overview", "Scatter Plot", "Model Evaluation"]
selected_page = st.sidebar.selectbox("Select a page", page_options)

if selected_page == "Data Overview":
    st.title("Data Overview")
    st.dataframe(iris.head())

elif selected_page == "Scatter Plot":
    st.title("Scatter Plot")

    selected_x_axes = st.multiselect("Select X Axes", iris.columns)

    if len(selected_x_axes) < 2:
        st.warning("Please select at least two X Axes.")

    else:
        y_axis = st.selectbox("Select Y Axis", iris.columns)

        fig, ax = plt.subplots(figsize=[10, 6])
        sns.scatterplot(data=iris, x=selected_x_axes[0], y=y_axis, hue="Species", style="Species")
        for x in selected_x_axes[1:]:
            sns.scatterplot(data=iris, x=x, y=y_axis, hue="Species", style="Species", ax=ax)
        st.pyplot(fig)

elif selected_page == "Model Evaluation":
    st.title("Model Evaluation")

    st.write("Column Names:", iris.columns)

    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    y = iris[['Species']].values

    model = LogisticRegression()
    model.fit(X, y)

    st.subheader("Model Evaluation:")
    st.write(f"Accuracy: {model.score(X, y)}")

    cm = metrics.confusion_matrix(y, model.predict(X))
    st.subheader("Confusion Matrix:")
    st.write(cm)
