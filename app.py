import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("DBSCAN Clustering (Visualization)")

# -----------------------------
# Sidebar: Student Info
# -----------------------------
logo_path = "images/parami.jpg"

st.sidebar.markdown("# **Mid term Group Project**")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

# st.sidebar.text("Thant Sin Tun")
# st.sidebar.text("Yin Min Han") 
# st.sidebar.text("Eh Si Si") 
# st.sidebar.text("Hnin Ei Wai Lwin") 
# st.sidebar.text("May Thiri Phyoe") 
# st.sidebar.text("May Thaw Tar") 
# st.sidebar.text("Peter Ling Mung") 

# -----------------------------
# Sidebar: Customer Inputs
# -----------------------------

st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", min_value=0, max_value=80, value=30)
income = st.sidebar.slider("Annual Income (k$)", min_value=0, max_value=150, value=50)
spending_score = st.sidebar.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])

# Map gender
gender_num = 1 if gender == 'Male' else 0

# user input dataframe
new_data = pd.DataFrame({
    'Gender': [gender_num],
    'Age': [age],
    'Annual Income (k$)': [income],
    'Spending Score (1-100)': [spending_score],
    'Gender (Male:1, Female:0)': [gender_num]
})

####################################################################################

# Load DBSCAN pipeline

with open('dbscan_pipeline.pkl', 'rb') as f:
    db_pipeline = pickle.load(f)

####################################################################################

# Load original data for visualization

org_df = pd.read_csv('Mall_Customers.csv')  

org_df['Gender'] = org_df['Gender'].map({'Male':1, 'Female':0})

X_orig = org_df[['Age', 'Annual Income (k$)','Spending Score (1-100)', 'Gender']]

# Transform original data
X_pca = db_pipeline['PCA'].transform(db_pipeline['preprocessing'].transform(X_orig))
labels_orig = db_pipeline['dbscan'].fit_predict(X_pca)

####################################################################################


# -----------------------------
# Train KNN on DBSCAN clusters
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_pca, labels_orig)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Cluster"):

    # Transform new customer
    transformed = db_pipeline['PCA'].transform(db_pipeline['preprocessing'].transform(new_data))

    # KNN nearest cluster
    distances, indices = knn.kneighbors(transformed)
    predicted_label = knn.predict(transformed)[0]

    # Apply distance threshold for noise
    eps = 0.35
    if distances[0][0] > eps:
        new_label = -1  # treat as noise
    else:
        new_label = predicted_label


    # Short descriptions for clusters
    cluster_descriptions = {
    -1: "Outliers: Mixed customers with varied age, income, and spending", # 19,, 16, 75
    0: "Young, Low income, High spending",
    1: "Middle-aged, Medium income, Moderate spending",
    2: "Young adults, Medium income, High spending",
    3: "Middle-aged, High income, Low spending"
    }

    # Display result
    st.subheader("Prediction Result:")
    if new_label == -1:
        st.warning("The new customer is considered NOISE (outlier).")
        st.info(f"{cluster_descriptions[new_label]}")
    else:
        st.success(f"The new customer belongs to Cluster {new_label}.")
        st.info(f"This Cluster belongs to {cluster_descriptions[new_label]} Customers.")

    st.subheader("Customer INPUT :")
    st.write(new_data)

    # -----------------------------
    # PCA Visualization
    # -----------------------------
    st.subheader("2D PCA Cluster Plot:")
    fig, ax = plt.subplots(figsize=(8,6))

    # Plot original clusters
    for lbl in set(labels_orig):
        if lbl == -1:
            color = 'grey'
            lbl_name = 'Noise'
        else:
            color = plt.cm.tab10(lbl % 10)
            lbl_name = f'Cluster {lbl}'
        ax.scatter(X_pca[labels_orig==lbl,0], X_pca[labels_orig==lbl,1], 
                   label=lbl_name, alpha=0.6)

    # Plot new customer
    ax.scatter(
        transformed[0,0], transformed[0,1],
        marker='X',
        s=150,
        color='red',
        edgecolor='black',
        label='New Customer'
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("DBSCAN Clusters (PCA 2D)")
    ax.legend()
    st.pyplot(fig)
