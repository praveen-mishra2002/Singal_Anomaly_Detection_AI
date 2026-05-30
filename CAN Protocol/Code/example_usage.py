"""
Quick Example: How to use the CAN Anomaly Detection System
"""

from detect_anomalies import CANAnomalyDetector
import pandas as pd

# Initialize detector with your data file
detector = CANAnomalyDetector('can_signal_data.csv')

# Step 1: Preprocess data (normalize)
detector.preprocess()

# Step 2: Train Isolation Forest model
detector.train_isolation_forest(contamination=0.05)

# Step 3: Train Local Outlier Factor model
detector.train_local_outlier_factor(n_neighbors=20, contamination=0.05)

# Step 4: Generate comprehensive report
results_df, detected_anomalies = detector.generate_report()

# Step 5: Create visualizations
detector.visualize_results()

# Access individual predictions
print("\nModel 1 (Isolation Forest) predictions:")
print(detector.predictions_if[:10])  # -1 = anomaly, 1 = normal

print("\nModel 2 (LOF) predictions:")
print(detector.predictions_lof[:10])  # -1 = anomaly, 1 = normal

# Access anomaly scores
print("\nAnomaly Scores (higher = more anomalous):")
print(detector.anomaly_scores_if[:10])

# Access results dataframe with all scores
print("\nTop 5 anomalies:")
print(detected_anomalies.head(5)[['timestamp', 'engine_rpm', 'engine_temp', 'relevance_score']])

# Filter high-confidence anomalies
_, relevance, _, _ = detector.calculate_relevance()
high_confidence = results_df[results_df['relevance_score'] > 0.7]
print(f"\nFound {len(high_confidence)} high-confidence anomalies (>0.7 relevance)")
