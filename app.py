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

st.sidebar.markdown("# **Group Members**")

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

# Function to assign nearest cluster

def assign_dbscan_cluster(new_point, X_pca, labels_orig, eps=0.35):
    clusters = np.unique(labels_orig[labels_orig != -1])
    min_dist = float('inf')
    assigned_cluster = -1
    
    for c in clusters:
        cluster_points = X_pca[labels_orig == c]
        dist = np.min(np.linalg.norm(cluster_points - new_point, axis=1))
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = c
    
    # Only assign if within eps distance
    if min_dist <= eps:
        return assigned_cluster
    else:
        return -1  # still considered noise

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Cluster"):

    # Transform new data
    transformed = db_pipeline['PCA'].transform(db_pipeline['preprocessing'].transform(new_data))

    # Assign cluster
    new_label = assign_dbscan_cluster(transformed[0], X_pca, labels_orig, eps=0.35)

    # Short descriptions for clusters
    cluster_descriptions = {
    -1: "Outliers: Mixed customers with varied age, income, and spending", # 19,, 16, 75
    0: "Young, Low income, High spending customers",
    1: "Middle-aged, Medium income, Moderate spending",
    2: "Young adults, Medium income, High spending",
    3: "Middle-aged, High income, Low spending"
    }

    # Display result
    st.subheader("Prediction Result:")
    if new_label == -1:
        st.warning("The new customer is considered NOISE (outlier).")
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
