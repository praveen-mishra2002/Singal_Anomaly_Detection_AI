import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent Code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Code'))

from detect_anomalies import CANAnomalyDetector
from generate_data import generate_can_signal_data
import os
import tempfile


class TestDataGeneration:
    """Test cases for synthetic CAN data generation."""

    def test_data_shape(self):
        """Test that generated data has correct shape."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=5)
        assert df.shape[0] == 100
        assert df.shape[1] == 10  # 8 signals + timestamp + is_anomaly

    def test_anomaly_percentage(self):
        """Test that anomaly percentage is approximately correct."""
        df = generate_can_signal_data(samples=1000, anomaly_percentage=10)
        actual_percentage = (df['is_anomaly'].sum() / len(df)) * 100
        assert 8 <= actual_percentage <= 12  # Allow 2% tolerance

    def test_required_columns(self):
        """Test that all required columns are present."""
        df = generate_can_signal_data(samples=50, anomaly_percentage=5)
        required_cols = ['timestamp', 'engine_rpm', 'vehicle_speed', 'throttle_position',
                         'engine_temp', 'brake_pressure', 'battery_voltage', 'fuel_level',
                         'oil_pressure', 'is_anomaly']
        assert all(col in df.columns for col in required_cols)

    def test_value_ranges(self):
        """Test that CAN signal values are in realistic ranges."""
        df = generate_can_signal_data(samples=500, anomaly_percentage=5)

        # Engine RPM: 0-8000
        assert df['engine_rpm'].min() >= 0
        assert df['engine_rpm'].max() <= 10000

        # Battery voltage: 10-14V
        assert df['battery_voltage'].min() >= 9
        assert df['battery_voltage'].max() <= 15

        # Engine temp: 70-130°C
        assert df['engine_temp'].min() >= 50
        assert df['engine_temp'].max() <= 150

    def test_timestamp_continuity(self):
        """Test that timestamps are continuous."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=5)
        time_diffs = df['timestamp'].diff()[1:]
        assert time_diffs.std() < pd.Timedelta('1ms')


class TestDataPreprocessing:
    """Test cases for data preprocessing."""

    def setup_method(self):
        """Setup test data."""
        self.df = generate_can_signal_data(samples=100, anomaly_percentage=5)
        self.detector = CANAnomalyDetector(None)
        self.detector.df = self.df

    def test_preprocessing_shapes(self):
        """Test preprocessing output shapes."""
        self.detector.preprocess()
        assert self.detector.X.shape == (100, 8)
        assert self.detector.X_scaled.shape == (100, 8)

    def test_scaling_normalization(self):
        """Test that scaling normalizes data to mean~0, std~1."""
        self.detector.preprocess()
        means = self.detector.X_scaled.mean(axis=0)
        stds = self.detector.X_scaled.std(axis=0)

        assert np.allclose(means, 0, atol=0.1)
        assert np.allclose(stds, 1, atol=0.1)

    def test_preprocessing_removes_nan(self):
        """Test handling of NaN values."""
        self.detector.preprocess()
        assert not np.isnan(self.detector.X_scaled).any()


class TestIsolationForest:
    """Test cases for Isolation Forest model."""

    def setup_method(self):
        """Setup detector with data."""
        self.df = generate_can_signal_data(samples=200, anomaly_percentage=10)
        self.detector = CANAnomalyDetector(None)
        self.detector.df = self.df
        self.detector.preprocess()

    def test_model_training(self):
        """Test that model trains without errors."""
        self.detector.train_isolation_forest()
        assert self.detector.model_if is not None

    def test_predictions_output(self):
        """Test prediction output format."""
        self.detector.train_isolation_forest()
        assert len(self.detector.predictions_if) == 200
        assert set(self.detector.predictions_if) == {-1, 1}

    def test_anomaly_scores(self):
        """Test anomaly scores are in valid range."""
        self.detector.train_isolation_forest()
        assert self.detector.anomaly_scores_if.min() >= 0
        assert self.detector.anomaly_scores_if.max() >= 0

    def test_contamination_parameter(self):
        """Test contamination parameter affects detection."""
        self.detector.train_isolation_forest(contamination=0.05)
        count_05 = (self.detector.predictions_if == -1).sum()

        self.detector.train_isolation_forest(contamination=0.15)
        count_15 = (self.detector.predictions_if == -1).sum()

        assert count_15 > count_05


class TestLocalOutlierFactor:
    """Test cases for LOF model."""

    def setup_method(self):
        """Setup detector with data."""
        self.df = generate_can_signal_data(samples=200, anomaly_percentage=10)
        self.detector = CANAnomalyDetector(None)
        self.detector.df = self.df
        self.detector.preprocess()

    def test_model_training(self):
        """Test that LOF model trains without errors."""
        self.detector.train_local_outlier_factor()
        assert self.detector.model_lof is not None

    def test_predictions_output(self):
        """Test LOF prediction output format."""
        self.detector.train_local_outlier_factor()
        assert len(self.detector.predictions_lof) == 200
        assert set(self.detector.predictions_lof) == {-1, 1}

    def test_anomaly_scores_normalized(self):
        """Test LOF scores are normalized to 0-1."""
        self.detector.train_local_outlier_factor()
        assert self.detector.anomaly_scores_lof.min() >= 0
        assert self.detector.anomaly_scores_lof.max() <= 1

    def test_n_neighbors_parameter(self):
        """Test n_neighbors parameter affects detection."""
        self.detector.train_local_outlier_factor(n_neighbors=10)
        predictions_10 = self.detector.predictions_lof.copy()

        self.detector.train_local_outlier_factor(n_neighbors=30)
        predictions_30 = self.detector.predictions_lof.copy()

        # Different n_neighbors should give different results
        assert not np.array_equal(predictions_10, predictions_30)


class TestRelevanceScoring:
    """Test cases for relevance score calculation."""

    def setup_method(self):
        """Setup detector with trained models."""
        self.df = generate_can_signal_data(samples=150, anomaly_percentage=10)
        self.detector = CANAnomalyDetector(None)
        self.detector.df = self.df
        self.detector.preprocess()
        self.detector.train_isolation_forest()
        self.detector.train_local_outlier_factor()

    def test_relevance_shape(self):
        """Test relevance scores have correct shape."""
        scores_if, relevance, agreement, _ = self.detector.calculate_relevance()
        assert len(relevance) == 150
        assert len(agreement) == 150

    def test_relevance_range(self):
        """Test relevance scores are between 0 and 1."""
        _, relevance, _, _ = self.detector.calculate_relevance()
        assert relevance.min() >= 0
        assert relevance.max() <= 1

    def test_agreement_binary(self):
        """Test agreement scores are binary (0 or 1)."""
        _, _, agreement, _ = self.detector.calculate_relevance()
        assert set(agreement) == {0, 1}

    def test_high_relevance_anomalies(self):
        """Test high relevance identifies strong anomalies."""
        _, relevance, agreement, _ = self.detector.calculate_relevance()

        high_rel = relevance > 0.7
        high_rel_agreement = agreement[high_rel]

        # High relevance should have more agreement
        if high_rel.sum() > 0:
            assert high_rel_agreement.mean() > 0.5


class TestModelIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test complete anomaly detection pipeline."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=5)
        detector = CANAnomalyDetector(None)
        detector.df = df

        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        results_df, detected_anomalies = detector.generate_report()

        assert len(results_df) == 100
        assert 'relevance_score' in results_df.columns
        assert len(detected_anomalies) > 0

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        df1 = generate_can_signal_data(samples=100, anomaly_percentage=5)
        df2 = generate_can_signal_data(samples=100, anomaly_percentage=5)

        # Should be identical since both use seed=42
        pd.testing.assert_frame_equal(df1, df2)

    def test_model_agreement_correlation(self):
        """Test that model agreement correlates with relevance."""
        df = generate_can_signal_data(samples=200, anomaly_percentage=10)
        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        _, relevance, agreement, _ = detector.calculate_relevance()

        # Agreement should correlate with relevance
        correlation = np.corrcoef(agreement, relevance)[0, 1]
        assert correlation > 0.3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test behavior with single sample."""
        df = generate_can_signal_data(samples=1, anomaly_percentage=0)
        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()

        assert detector.X_scaled.shape[0] == 1

    def test_small_dataset(self):
        """Test with very small dataset."""
        df = generate_can_signal_data(samples=10, anomaly_percentage=0)
        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        assert detector.predictions_if is not None

    def test_all_anomalies(self):
        """Test when all samples are marked as anomalies."""
        df = generate_can_signal_data(samples=50, anomaly_percentage=100)
        assert df['is_anomaly'].sum() > 0

    def test_no_anomalies(self):
        """Test when no samples are marked as anomalies."""
        df = generate_can_signal_data(samples=50, anomaly_percentage=0)
        assert df['is_anomaly'].sum() == 0

    def test_extreme_values(self):
        """Test handling of extreme signal values."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=20)

        # Manually add extreme values
        df.loc[0, 'engine_rpm'] = 9000
        df.loc[1, 'battery_voltage'] = 6.0

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should detect extremes as anomalies
        assert detector.predictions_if[0] == -1 or detector.predictions_if[1] == -1


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_results_dataframe_structure(self):
        """Test that results dataframe has all required columns."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=5)
        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        results_df, _ = detector.generate_report()

        required_cols = [
            'anomaly_score_if', 'anomaly_score_lof', 'combined_anomaly_score',
            'relevance_score', 'model_agreement', 'prediction_if', 'prediction_lof'
        ]

        assert all(col in results_df.columns for col in required_cols)

    def test_detected_anomalies_sorted(self):
        """Test that detected anomalies are sorted by relevance."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=10)
        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        _, detected = detector.generate_report()

        if len(detected) > 1:
            relevance_scores = detected['relevance_score'].values
            assert all(relevance_scores[i] >= relevance_scores[i+1]
                      for i in range(len(relevance_scores)-1))


class TestDataPersistence:
    """Test file I/O operations."""

    def test_csv_write_read(self):
        """Test CSV write and read operations."""
        df = generate_can_signal_data(samples=50, anomaly_percentage=5)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name

        try:
            df.to_csv(temp_file, index=False)
            df_read = pd.read_csv(temp_file)

            assert len(df_read) == 50
            assert 'timestamp' in df_read.columns
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_results_output_format(self):
        """Test that results CSV is properly formatted."""
        df = generate_can_signal_data(samples=50, anomaly_percentage=5)
        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        results_df, _ = detector.generate_report()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name

        try:
            results_df.to_csv(temp_file, index=False)
            results_read = pd.read_csv(temp_file)

            assert len(results_read) == len(results_df)
            assert results_read['relevance_score'].dtype in [np.float64, np.float32]
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
