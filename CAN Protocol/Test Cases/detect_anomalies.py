import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class CANAnomalyDetector:
    """CAN Signal Anomaly Detection using two different models."""

    def __init__(self, data_file):
        self.df = pd.read_csv(data_file) if data_file else None
        self.scaler = StandardScaler()
        self.feature_cols = [
            'engine_rpm', 'vehicle_speed', 'throttle_position',
            'engine_temp', 'brake_pressure', 'battery_voltage',
            'fuel_level', 'oil_pressure'
        ]
        self.X = None
        self.X_scaled = None
        self.model_if = None
        self.model_lof = None
        self.predictions_if = None
        self.predictions_lof = None
        self.anomaly_scores_if = None
        self.anomaly_scores_lof = None

    def preprocess(self):
        """Preprocess and normalize the data."""
        print("[INFO] Preprocessing data...")
        self.X = self.df[self.feature_cols].values
        self.X_scaled = self.scaler.fit_transform(self.X)
        print(f"[OK] Scaled {self.X_scaled.shape[0]} samples with {self.X_scaled.shape[1]} features")

    def train_isolation_forest(self, contamination=0.05):
        """Train Isolation Forest model."""
        print("\n[INFO] Training Isolation Forest...")
        self.model_if = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.predictions_if = self.model_if.fit_predict(self.X_scaled)
        self.anomaly_scores_if = -self.model_if.score_samples(self.X_scaled)  # Higher = more anomalous

        n_anomalies_if = (self.predictions_if == -1).sum()
        print(f"[OK] Detected {n_anomalies_if} anomalies")

    def train_local_outlier_factor(self, n_neighbors=20, contamination=0.05):
        """Train Local Outlier Factor model."""
        print("[INFO] Training Local Outlier Factor...")
        self.model_lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        self.predictions_lof = self.model_lof.fit_predict(self.X_scaled)
        self.anomaly_scores_lof = -self.model_lof.negative_outlier_factor_  # Higher = more anomalous
        self.anomaly_scores_lof = (self.anomaly_scores_lof - self.anomaly_scores_lof.min()) / \
                                  (self.anomaly_scores_lof.max() - self.anomaly_scores_lof.min())

        n_anomalies_lof = (self.predictions_lof == -1).sum()
        print(f"[OK] Detected {n_anomalies_lof} anomalies")

    def calculate_relevance(self):
        """Calculate relevance/confidence scores combining both models."""
        print("\n[INFO] Calculating relevance scores...")

        # Normalize Isolation Forest scores (0-1)
        scores_if_norm = (self.anomaly_scores_if - self.anomaly_scores_if.min()) / \
                         (self.anomaly_scores_if.max() - self.anomaly_scores_if.min())

        # Combine scores: average of both models
        combined_score = (scores_if_norm + self.anomaly_scores_lof) / 2

        # Agreement between models
        agreement = ((self.predictions_if == -1) & (self.predictions_lof == -1)).astype(int)

        # Relevance: combination of combined score and model agreement
        relevance = combined_score * (0.7) + agreement * (0.3)

        return combined_score, relevance, agreement, scores_if_norm

    def generate_report(self):
        """Generate comprehensive anomaly detection report."""
        print("\n" + "="*70)
        print("CAN SIGNAL ANOMALY DETECTION REPORT")
        print("="*70)

        scores_if_norm, relevance, agreement, _ = self.calculate_relevance()

        # Create results dataframe
        results_df = self.df.copy()
        results_df['anomaly_score_if'] = self.anomaly_scores_if
        results_df['anomaly_score_lof'] = self.anomaly_scores_lof
        results_df['combined_anomaly_score'] = scores_if_norm
        results_df['relevance_score'] = relevance
        results_df['model_agreement'] = agreement
        results_df['prediction_if'] = self.predictions_if
        results_df['prediction_lof'] = self.predictions_lof

        # Detected anomalies
        detected_anomalies = results_df[
            (results_df['prediction_if'] == -1) | (results_df['prediction_lof'] == -1)
        ].copy()
        detected_anomalies = detected_anomalies.sort_values('relevance_score', ascending=False)

        print(f"\n[STATS] DETECTION STATISTICS:")
        print("---" * 70)
        print(f"Total samples analyzed: {len(results_df)}")
        print(f"Isolation Forest anomalies: {(self.predictions_if == -1).sum()}")
        print(f"Local Outlier Factor anomalies: {(self.predictions_lof == -1).sum()}")
        print(f"Both models agree: {agreement.sum()}")
        print(f"High relevance anomalies (>0.7): {(relevance > 0.7).sum()}")

        # Performance metrics if ground truth available
        if 'is_anomaly' in results_df.columns:
            print(f"\n[PERFORMANCE] MODEL PERFORMANCE:")
            print(f"{'---'*70}")
            y_true = results_df['is_anomaly'].values

            # Isolation Forest metrics
            print(f"\nIsolation Forest:")
            print(classification_report(y_true, self.predictions_if == -1,
                                        target_names=['Normal', 'Anomaly'], digits=3))

            # LOF metrics
            print(f"Local Outlier Factor:")
            print(classification_report(y_true, self.predictions_lof == -1,
                                        target_names=['Normal', 'Anomaly'], digits=3))

            # Ensemble metrics
            ensemble_pred = ((self.predictions_if == -1) | (self.predictions_lof == -1)).astype(int)
            print(f"Ensemble (either model):")
            print(classification_report(y_true, ensemble_pred,
                                        target_names=['Normal', 'Anomaly'], digits=3))

        print(f"\n[ANOMALIES] TOP 10 DETECTED ANOMALIES (by relevance):")
        print(f"{'---'*70}")
        if len(detected_anomalies) > 0:
            display_cols = ['timestamp', 'engine_rpm', 'engine_temp', 'battery_voltage',
                           'anomaly_score_if', 'anomaly_score_lof', 'relevance_score']
            print(detected_anomalies[display_cols].head(10).to_string(index=False))
        else:
            print("No anomalies detected")

        # Save detailed results
        results_df.to_csv('anomaly_detection_results.csv', index=False)
        print(f"\n[OK] Detailed results saved to: anomaly_detection_results.csv")

        return results_df, detected_anomalies

    def visualize_results(self):
        """Create visualization plots."""
        print("\n[INFO] Generating visualizations...")

        scores_if_norm, relevance, agreement, _ = self.calculate_relevance()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('CAN Signal Anomaly Detection Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Anomaly Scores Comparison
        ax = axes[0, 0]
        sample_range = range(min(200, len(self.anomaly_scores_if)))
        ax.plot(sample_range, self.anomaly_scores_if[sample_range], label='Isolation Forest', alpha=0.7)
        ax.plot(sample_range, self.anomaly_scores_lof[sample_range], label='LOF', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores Over Time (First 200 samples)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Distribution of Anomaly Scores
        ax = axes[0, 1]
        normal_if = scores_if_norm[self.predictions_if == 1]
        anomaly_if = scores_if_norm[self.predictions_if == -1]
        ax.hist(normal_if, bins=30, alpha=0.6, label='Normal (IF)', color='blue')
        ax.hist(anomaly_if, bins=30, alpha=0.6, label='Anomaly (IF)', color='red')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Anomaly Scores - Isolation Forest')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Model Agreement
        ax = axes[1, 0]
        agreement_counts = [
            (agreement == 0).sum(),
            (agreement == 1).sum()
        ]
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(agreement_counts, labels=['Disagreement', 'Agreement'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Model Agreement on Anomalies')

        # Plot 4: Relevance Score Distribution
        ax = axes[1, 1]
        ax.hist(relevance, bins=40, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(relevance.mean(), color='red', linestyle='--', label=f'Mean: {relevance.mean():.3f}')
        ax.axvline(0.7, color='green', linestyle='--', label='High Relevance Threshold')
        ax.set_xlabel('Relevance Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Relevance Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('anomaly_detection_visualization.png', dpi=300, bbox_inches='tight')
        print("[OK] Visualization saved to: anomaly_detection_visualization.png")
        plt.close()

def main():
    """Main execution."""
    data_file = 'can_signal_data.csv'

    # Check if data file exists, generate if not
    import os
    if not os.path.exists(data_file):
        print("[INFO] Data file not found. Generating synthetic data...")
        from generate_data import generate_can_signal_data
        df = generate_can_signal_data(samples=1000, anomaly_percentage=5)
        df.to_csv(data_file, index=False)
        print(f"[OK] Generated {len(df)} samples")

    # Initialize detector
    detector = CANAnomalyDetector(data_file)

    # Process
    detector.preprocess()
    detector.train_isolation_forest()
    detector.train_local_outlier_factor()
    results_df, detected_anomalies = detector.generate_report()
    detector.visualize_results()

    print("\n[COMPLETE] Analysis complete!")
    print("Output files:")
    print("  - anomaly_detection_results.csv")
    print("  - anomaly_detection_visualization.png")

if __name__ == '__main__':
    main()
