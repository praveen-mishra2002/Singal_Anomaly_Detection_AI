# CAN Signal Anomaly Detection AI
# 30-May-2026

A comprehensive anomaly detection system for vehicle CAN (Controller Area Network) signals using two different machine learning models.

## 📋 Overview

This project detects anomalies in automotive CAN signals by employing:

1. **Isolation Forest** - An ensemble method that isolates anomalies by randomly selecting features
2. **Local Outlier Factor (LOF)** - A density-based approach that identifies local density deviations

The system combines predictions from both models with relevance scoring to provide high-confidence anomaly detection.

## 🎯 Features

- **Synthetic Data Generation**: Realistic CAN signal patterns with controlled anomalies
- **Dual Model Approach**: Combines two different anomaly detection algorithms
- **Relevance Scoring**: Confidence scores based on model agreement and individual scores
- **Performance Metrics**: Precision, recall, F1-score, and confusion matrices
- **Visualizations**: Comprehensive plots showing anomaly patterns and model comparison
- **Detailed Reporting**: CSV output with all predictions and scores

## 📊 CAN Signals Analyzed

The system monitors these vehicle signals:
- Engine RPM
- Vehicle Speed
- Throttle Position
- Engine Temperature
- Brake Pressure
- Battery Voltage
- Fuel Level
- Oil Pressure

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python generate_data.py
```
Output: `can_signal_data.csv` (1000 samples with 5% anomalies)

### 3. Run Anomaly Detection
```bash
python detect_anomalies.py
```

## 📁 Output Files

- **anomaly_detection_results.csv**: Complete results with scores for each sample
  - `anomaly_score_if`: Isolation Forest score (0-1)
  - `anomaly_score_lof`: LOF score (0-1)
  - `combined_anomaly_score`: Average of both models
  - `relevance_score`: Final confidence score (0-1)
  - `model_agreement`: 1 if both models agree, 0 otherwise

- **anomaly_detection_visualization.png**: Four-panel visualization

## 🔍 Two-Model Approach

### Model 1: Isolation Forest
- **What it does**: Uses ensemble of random trees to isolate anomalies
- **Best for**: Global anomalies (extreme values far from normal)
- **Speed**: Very fast, scales well
- **Strength**: Works with high-dimensional data

### Model 2: Local Outlier Factor (LOF)
- **What it does**: Compares local density of points to neighbors
- **Best for**: Contextual anomalies (unusual in local context)
- **Speed**: Moderate, depends on neighborhood size
- **Strength**: Detects subtle, localized deviations

## 📈 Relevance Score

```
Relevance Score = (Combined Score × 0.7) + (Model Agreement × 0.3)
```

- **0.0-0.3**: Low confidence anomaly
- **0.3-0.7**: Medium confidence
- **0.7-1.0**: High confidence (both models agree)

## 🚗 Anomaly Types Detected

1. **High-value anomalies**: Engine RPM spikes, temperature increases
2. **Low-value anomalies**: Battery voltage drops, oil pressure loss
3. **Spike anomalies**: Sudden throttle changes, RPM fluctuations
4. **Pattern anomalies**: Unusual brake pressure with low speed

## 📝 Example Output

```
CAN SIGNAL ANOMALY DETECTION REPORT
==============================================================================

📈 DETECTION STATISTICS:
────────────────────────────────────────────────────────────────────────────
Total samples analyzed: 1000
Isolation Forest anomalies: 50
Local Outlier Factor anomalies: 48
Both models agree: 45
High relevance anomalies (>0.7): 42

🎯 MODEL PERFORMANCE:
────────────────────────────────────────────────────────────────────────────
Isolation Forest:
              precision    recall  f1-score
Normal           0.980    0.985    0.982
Anomaly          0.824    0.760    0.791

Local Outlier Factor:
              precision    recall  f1-score
Normal           0.984    0.980    0.982
Anomaly          0.838    0.800    0.818

🔴 TOP DETECTED ANOMALIES:
────────────────────────────────────────────────────────────────────────────
     timestamp  engine_rpm  engine_temp  battery_voltage  relevance_score
...
```

## 🔧 Configuration

Edit parameters in `detect_anomalies.py`:

```python
# Adjust contamination (expected anomaly percentage)
detector.train_isolation_forest(contamination=0.05)

# Adjust LOF neighborhood size
detector.train_local_outlier_factor(n_neighbors=20)
```

## 💡 Tips & Tricks

**To reduce false positives:**
- Increase `n_neighbors` in LOF
- Decrease `contamination` parameter
- Require higher `relevance_score` threshold

**To catch more anomalies:**
- Decrease `n_neighbors` in LOF
- Increase `contamination` parameter
- Lower `relevance_score` threshold

**For production use:**
- Always validate with ground truth data
- Monitor detection rates over time
- Adjust thresholds based on domain knowledge

## 📊 Files Included

- `generate_data.py` - Synthetic CAN signal data generator
- `detect_anomalies.py` - Main anomaly detection pipeline
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎓 Educational Resources

- Isolation Forest: Isolates anomalies through random subsampling
- LOF: Measures local reachability density of each point
- Combined approach: Leverages strengths of both algorithms

## ✅ Next Steps

1. Run the complete pipeline: `python detect_anomalies.py`
2. Review results in `anomaly_detection_results.csv`
3. Analyze visualizations in `anomaly_detection_visualization.png`
4. Adapt thresholds based on your use case
5. Deploy to production with your real CAN data

---

**Created for automotive anomaly detection and predictive maintenance**
