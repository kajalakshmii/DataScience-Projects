import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv("D:\\internship\\project in regression\\st_project_simple linear regression\\car_dataset.csv")

st.title("Linear Regression App")

st.sidebar.title("Car Dataset")
page_options = ["Data Overview", "Simple Linear Regression", "Multiple Linear Regression"]
selected_page = st.sidebar.selectbox("Select a page", page_options)

if selected_page == "Data Overview":
    st.header("Data Overview")
    st.dataframe(df.head())

elif selected_page == "Simple Linear Regression":
    st.header("Simple Linear Regression")

    feature = st.selectbox("Select feature for X-axis", df.columns)
    target = st.selectbox("Select target variable for Y-axis", df.columns)

    x = df[feature]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(x_train.values.reshape(-1, 1), y_train)
    y_pred = model.predict(x_test.values.reshape(-1, 1))

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R2: {r2}")

    st.subheader("Scatter plot of Actual vs Predicted Values")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

elif selected_page == "Multiple Linear Regression":
    st.header("Multiple Linear Regression")

    features_multiple = st.multiselect("Select features for X-axis", df.columns)
    target_multiple = st.selectbox("Select target variable for Y-axis", df.columns)  # Add target selection
    x_multiple = df[features_multiple]
    y_multiple = df[target_multiple]  

    model_multiple = LinearRegression()
    model_multiple.fit(x_multiple, y_multiple)
    y_pred_multiple = model_multiple.predict(x_multiple)

    mse_multiple = mean_squared_error(y_multiple, y_pred_multiple)
    rmse_multiple = math.sqrt(mse_multiple)
    mae_multiple = mean_absolute_error(y_multiple, y_pred_multiple)
    r2_multiple = r2_score(y_multiple, y_pred_multiple)

    st.subheader("Model Evaluation (Multiple Linear Regression)")
    st.write(f"MSE: {mse_multiple}")
    st.write(f"RMSE: {rmse_multiple}")
    st.write(f"MAE: {mae_multiple}")
    st.write(f"R2: {r2_multiple}")

    st.subheader("Scatter plot of Actual vs Predicted Values (Multiple Linear Regression)")
    fig_multiple, ax_multiple = plt.subplots()
    sns.scatterplot(x=y_multiple, y=y_pred_multiple, ax=ax_multiple)
    ax_multiple.set_xlabel("Actual")
    ax_multiple.set_ylabel("Predicted")
    st.pyplot(fig_multiple)
