
# Anomaly Detection

## Overview

This project develops an AI-based anomaly detection system to identify unusual operating patterns in industrial machine sensor data.
The system uses the Isolation Forest algorithm to distinguish normal observations from potential anomalies.

## Problem Statement

Industrial machines continuously generate sensor data such as temperature, rotational speed, torque, and tool wear.
Unusual sensor patterns may indicate abnormal operating conditions that require further investigation.
This project explores how unsupervised machine learning can support industrial condition monitoring.

## Dataset

AI4I 2020 Predictive Maintenance Dataset.

Source:
https://archive.ics.uci.edu/dataset/601/

The dataset contains 10,000 observations of machine
operational parameters.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Isolation Forest
- StandardScaler
- Matplotlib
- Streamlit

## Methodology

1. Load industrial sensor data.
2. Select relevant sensor features.
3. Check missing values.
4. Standardize numerical features.
5. Train an Isolation Forest model.
6. Detect anomalous observations.
7. Visualize normal and abnormal patterns.
8. Export detection results.

## Algorithm

Isolation Forest.
The algorithm identifies unusual observations by isolating data points that differ from the majority.

## Features

- CSV dataset upload.
- Configurable anomaly contamination.
- Sensor anomaly visualization.
- Normal and anomaly counts.
- Anomaly detection rate.
- Downloadable detection results.

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Important Notes

The contamination parameter represents an assumed anomaly proportion used by the model. It does not represent the actual machine failure rate.
Anomaly detection results are indicators for further inspection, not confirmed equipment failures.

## Future Improvements

- Compare anomaly results with actual failure labels.
- Evaluate detection precision and recall.
- Add real-time sensor data integration.
- Develop automated industrial alerts.

## Author
Nabilah Robbaniyah — Informatics Student
