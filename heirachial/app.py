import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

df = pd.read_csv('D:\\internship\\project in regression\\heirachial\\CC GENERAL.csv')
df = df.drop('CUST_ID', axis=1)
df.fillna(method='ffill', inplace=True)
page_options = ["Data Overview", "Dendrogram", "Silhouette Analysis", "Clustering Result"]
selected_page = st.sidebar.selectbox("Select a page", page_options)

if selected_page == "Data Overview":
    st.title("Data Overview")
    st.dataframe(df.head())

elif selected_page == "Dendrogram":
    st.title("Dendrogram")
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    normalized_df = normalize(scaled_df)
    normalized_df = pd.DataFrame(normalized_df)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(normalized_df)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(X_principal, method='ward')))
    st.pyplot(fig)

elif selected_page == "Silhouette Analysis":
    st.title("Silhouette Analysis")
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    normalized_df = normalize(scaled_df)
    normalized_df = pd.DataFrame(normalized_df)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(normalized_df)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    silhouette_scores = []
    for n_cluster in range(2, 8):
        silhouette_scores.append(
            silhouette_score(X_principal, AgglomerativeClustering(n_clusters=n_cluster).fit_predict(X_principal)))
    k = [2, 3, 4, 5, 6, 7]
    fig, ax = plt.subplots()
    ax.bar(k, silhouette_scores)
    ax.set_xlabel('Number of clusters', fontsize=10)
    ax.set_ylabel('Silhouette Score', fontsize=10)
    st.pyplot(fig)

elif selected_page == "Clustering Result":
    st.title("Clustering Result")
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    normalized_df = normalize(scaled_df)
    normalized_df = pd.DataFrame(normalized_df)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(normalized_df)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    agg = AgglomerativeClustering(n_clusters=2)  
    cluster_labels = agg.fit_predict(X_principal)
    fig, ax = plt.subplots()
    ax.scatter(X_principal['P1'], X_principal['P2'], c=cluster_labels, cmap=plt.cm.winter)
    st.pyplot(fig)
