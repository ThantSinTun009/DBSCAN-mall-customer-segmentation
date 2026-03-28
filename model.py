import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
import pickle

# Read
df = pd.read_csv('Mall_Customers.csv')

# Clean
df.drop('CustomerID', axis = 1, inplace = True)

# Encode
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

org_df = df.copy()

# Pipeline
numerical = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

preprocessor = ColumnTransformer([
    ("Scaler", StandardScaler(), numerical) # Scale
    ])

db_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('PCA', PCA(n_components=2)), # PCA
    ('dbscan', DBSCAN(eps=0.35, min_samples=10))
])

# Fit pipeline
db_pipeline.fit(org_df)

labels = db_pipeline['dbscan'].labels_

# PCA df
X_pca = db_pipeline['PCA'].transform(
    db_pipeline['preprocessing'].transform(org_df)
)

# Remove noise (masking)
mask = labels != -1
df_clean = X_pca[mask]
labels_clean = labels[mask]

score = silhouette_score(df_clean, labels_clean)
print("Silhouette score:", round(score, 3))

# Count noise
noise_points = np.sum(labels == -1)
print(f"Number of noise points: {noise_points}")

# Save model
with open('dbscan_pipeline.pkl', 'wb') as f:
    pickle.dump(db_pipeline, f)
    
print("Saved Model Successfully...")
