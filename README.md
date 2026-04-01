# Mall Customer Segmentation using DBSCAN

## Overview

This project performs **Customer segmentation** using the **DBSCAN clustering algorithm** on the Mall Customers dataset.

The goal is to group customers based on their:

* Gender
* Age
* Annual Income (k$)
* Spending Score (1–100)

We also compare DBSCAN with KMeans using **Silhouette Score** to evaluate clustering quality.

**Streamlit Demo:** https://dbscan-mall-customer-segmentation.streamlit.app/ 

---

## Project Structure

```bash
DBSCAN-mall-customer-segmentation/

├── notebook.ipynb              # Analysis, Preprocessing, PCA, Comparison with KMeans

├── model.py                    # Extract final model (DBSCAN pipeline)
├── dbscan_pipeline.pkl         # Saved trained pipeline
├── app.py                      # Streamlit demonstration
├── Mall_Customers.csv          # Dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---


## Usage Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the model

```bash
python model.py
```

### 3. Run the Streamlit

```bash
streamlit run app.py
```

---

## Group Members

- Eh Si Si
- Peter Ling Mung
- May Thaw Tar
- May Thiri Phyoe
- Yin Min Han
- Thant Sin Tun
