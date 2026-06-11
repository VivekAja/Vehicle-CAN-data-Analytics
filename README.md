# Vehicle CAN Analytics: Predictive Maintenance & Anomaly Detection

## Overview

This project demonstrates an end-to-end data analytics and machine learning pipeline for vehicle CAN (Controller Area Network) telemetry data. The objective is to analyze vehicle subsystem behavior, identify patterns associated with failures, and detect anomalous operating conditions that may indicate maintenance requirements or system faults.

The project covers:

* Exploratory Data Analysis (EDA)
* Feature relationship analysis
* Failure pattern identification
* Anomaly detection using machine learning
* Visualization of subsystem health indicators

---

## Project Structure

```text
vehicle-can-analytics/
│
├── data/
│   ├── synthetic_telemetry_data.csv
│   ├── df_eda.csv
│   └── df_anomaly.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_anomaly.ipynb
│   └── 03_ml.ipynb
│
├── outputs/
│   ├── 01_target_distribution.png
│   ├── 02_engine_distributions.png
│   ├── ...
│   └── 15_multi_subsystem_cooccurrence.png
│
├── src/
│   ├── predict.py
│   ├── simulate.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses synthetic CAN telemetry data representing multiple vehicle subsystems, including:

* Engine
* Braking System
* Wheel Speed Sensors
* Battery System
* Vehicle Operating Conditions

The dataset contains sensor readings and target indicators that simulate component degradation and failure scenarios.

---

## Exploratory Data Analysis

The EDA phase focuses on:

* Target class distributions
* Sensor value distributions
* Outlier detection
* Correlation analysis
* Time-series behavior
* Brand-level failure analysis

### Key Visualizations

* Target Distribution
* Engine Sensor Analysis
* Brake System Analysis
* Battery Health Analysis
* Feature Correlation Matrix
* Vehicle Time-Series Trends

---

## Anomaly Detection

The anomaly detection pipeline identifies unusual operating conditions that may indicate:

* Component malfunction
* Sensor abnormalities
* Early warning signs of failure
* Multi-subsystem fault interactions

Techniques include:

* Feature engineering
* Statistical anomaly scoring
* Machine learning–based anomaly detection
* Performance evaluation using confusion matrices

### Outputs

* Anomaly score distributions
* Confusion matrices
* Time-series anomaly overlays
* Violin plots for anomaly separation
* Multi-subsystem anomaly co-occurrence analysis

---

## Technology Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/VivekAja/Vehicle-CAN-data-Analytics.git
cd Vehicle-CAN-data-Analytics
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run notebooks in the following order:

1. `01_eda.ipynb`
2. `02_anomaly.ipynb`
3. `03_ml.ipynb`

---

Generated visualizations will be saved in the `outputs/` directory.

---

## Business Value

Vehicle telemetry analytics can help organizations:

* Reduce unexpected breakdowns
* Improve fleet reliability
* Enable predictive maintenance
* Lower maintenance costs
* Improve vehicle safety
* Increase operational efficiency

---

## Future Improvements

* Real-world CAN bus data integration
* Streaming anomaly detection
* Predictive failure forecasting
* Dashboard deployment using Streamlit
* Model monitoring and retraining pipeline
* Cloud deployment for fleet-scale monitoring

---

## Author

Vivek A

Machine Learning | Data Science | Mechanical Engineering
