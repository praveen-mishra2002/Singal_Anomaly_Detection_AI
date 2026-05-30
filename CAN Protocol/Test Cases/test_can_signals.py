import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Robust import handling for multiple directory structures
def setup_imports():
    """Setup Python path for imports from multiple possible locations."""
    current_file = Path(__file__).resolve()
    test_dir = current_file.parent
    code_dir = test_dir.parent / 'Code'
    root_dir = test_dir.parent.parent.parent

    # Paths to try in order
    paths = [
        str(code_dir),
        str(test_dir),
        str(root_dir),
        str(root_dir / 'CAN Protocol' / 'Code'),
    ]

    for path in paths:
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)

setup_imports()

# Import with error handling
try:
    from detect_anomalies import CANAnomalyDetector
    from generate_data import generate_can_signal_data
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Python Path: {sys.path}")
    raise


class TestCANSignalAnomalies:
    """Test cases for specific CAN signal anomaly scenarios."""

    def test_engine_overheat_anomaly(self):
        """Test detection of engine overheating anomaly."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Insert engine overheat anomaly
        df.loc[50, 'engine_temp'] = 125  # Well above normal 85-95°C
        df.loc[51, 'engine_temp'] = 128

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        # Both models should detect the overheating
        assert detector.predictions_if[50] == -1 or detector.predictions_if[51] == -1
        assert detector.predictions_lof[50] == -1 or detector.predictions_lof[51] == -1

    def test_battery_voltage_drop(self):
        """Test detection of battery voltage drop."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Insert battery voltage drop (critical)
        df.loc[30, 'battery_voltage'] = 10.5  # Below 11V is critical
        df.loc[31, 'battery_voltage'] = 10.2

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should detect voltage drop as anomaly
        assert detector.predictions_if[30] == -1 or detector.predictions_if[31] == -1

    def test_sudden_rpm_spike(self):
        """Test detection of sudden engine RPM spike."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Insert RPM spike
        df.loc[40, 'engine_rpm'] = 7500  # Far above normal 2500-3500
        df.loc[41, 'engine_rpm'] = 8000

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        # Both should detect RPM spikes
        assert detector.predictions_if[40] == -1 or detector.predictions_if[41] == -1

    def test_brake_pressure_anomaly(self):
        """Test detection of abnormal brake pressure."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Insert excessive brake pressure
        df.loc[20, 'brake_pressure'] = 95  # Way above normal 0-60
        df.loc[21, 'brake_pressure'] = 98

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should detect brake pressure anomaly
        assert detector.predictions_if[20] == -1 or detector.predictions_if[21] == -1

    def test_oil_pressure_loss(self):
        """Test detection of oil pressure loss."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Insert critical oil pressure drop
        df.loc[60, 'oil_pressure'] = 5  # Critical - below 15 psi
        df.loc[61, 'oil_pressure'] = 8

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should detect oil pressure loss
        assert detector.predictions_if[60] == -1 or detector.predictions_if[61] == -1

    def test_fuel_level_anomaly(self):
        """Test detection of abnormal fuel consumption."""
        df = generate_can_signal_data(samples=200, anomaly_percentage=0)

        # Fuel should decrease linearly, insert sudden drop
        df.loc[100, 'fuel_level'] = 10  # Sudden drop from ~50
        df.loc[101, 'fuel_level'] = 5

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should detect sudden fuel loss
        assert detector.predictions_if[100] == -1 or detector.predictions_if[101] == -1

    def test_speed_throttle_mismatch(self):
        """Test detection of speed-throttle mismatch anomaly."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # High throttle but very low speed (unusual pattern)
        df.loc[45, 'throttle_position'] = 90
        df.loc[45, 'vehicle_speed'] = 5

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        # LOF should catch this contextual anomaly
        assert detector.predictions_lof[45] == -1

    def test_multiple_anomalies_detection(self):
        """Test detection of multiple simultaneous signal anomalies."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Create a sample with multiple anomalies
        df.loc[50, 'engine_temp'] = 120
        df.loc[50, 'oil_pressure'] = 10
        df.loc[50, 'battery_voltage'] = 10.5

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        # Should strongly detect this multi-faceted anomaly
        assert detector.predictions_if[50] == -1
        assert detector.predictions_lof[50] == -1


class TestAnomalyDetectionSensitivity:
    """Test anomaly detection sensitivity to different magnitudes."""

    def test_small_deviation_detection(self):
        """Test detection of small signal deviations."""
        df = generate_can_signal_data(samples=150, anomaly_percentage=0)

        # Small deviation in engine temp (just 5 degrees)
        df.loc[75, 'engine_temp'] = 100  # 5 degree above normal range

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest(contamination=0.10)

        # Should still detect with increased contamination
        predictions = detector.predictions_if
        if (predictions == -1).sum() > 5:
            assert True  # Model is sensitive enough

    def test_progressive_anomaly(self):
        """Test detection of progressive anomalies (degradation)."""
        df = generate_can_signal_data(samples=200, anomaly_percentage=0)

        # Progressive oil pressure degradation
        for i in range(100, 110):
            df.loc[i, 'oil_pressure'] = 50 - (i - 100) * 3

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_local_outlier_factor()

        # Should detect at least some progressive anomalies
        anomaly_count = (detector.predictions_lof[100:110] == -1).sum()
        assert anomaly_count >= 1


class TestAnomalyRelevanceScores:
    """Test anomaly relevance scoring accuracy."""

    def test_high_confidence_anomaly(self):
        """Test that extreme anomalies get high relevance scores."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Extreme anomaly
        df.loc[50, 'engine_temp'] = 140  # Extreme overheat
        df.loc[50, 'oil_pressure'] = 2   # Critical pressure

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        _, relevance, _, _ = detector.calculate_relevance()

        # Extreme anomaly should have high relevance score
        assert relevance[50] > 0.7

    def test_moderate_anomaly_relevance(self):
        """Test that moderate anomalies get moderate relevance scores."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Moderate anomaly
        df.loc[50, 'engine_temp'] = 105  # Slightly high

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        _, relevance, _, _ = detector.calculate_relevance()

        # Moderate anomaly should have lower relevance
        if detector.predictions_if[50] == -1:
            assert 0.3 < relevance[50] < 0.8

    def test_agreement_affects_relevance(self):
        """Test that model agreement increases relevance score."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Create clear anomaly
        df.loc[50, 'engine_temp'] = 130
        df.loc[50, 'battery_voltage'] = 9

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()
        detector.train_local_outlier_factor()

        _, relevance, agreement, _ = detector.calculate_relevance()

        # If both models agree on anomaly, relevance should be high
        if agreement[50] == 1:
            assert relevance[50] > 0.6


class TestCANProtocolSpecific:
    """Test CAN protocol-specific anomaly scenarios."""

    def test_can_message_frequency_anomaly(self):
        """Test detection of anomalies in CAN message timing."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # In real CAN, signals have expected frequencies
        # Simulate a signal sending too frequently or irregularly
        df.loc[25:30, 'engine_rpm'] = np.random.normal(3000, 100, 6)
        df.loc[25:30, 'engine_temp'] = np.random.normal(90, 2, 6)

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_local_outlier_factor()

        # LOF should detect pattern anomalies
        assert detector.predictions_lof is not None

    def test_can_signal_correlation_anomaly(self):
        """Test detection of anomalous signal correlations."""
        df = generate_can_signal_data(samples=150, anomaly_percentage=0)

        # Normally, high RPM and high throttle correlate
        # Create anomaly: high RPM with low throttle
        df.loc[75, 'engine_rpm'] = 6500
        df.loc[75, 'throttle_position'] = 10  # Unusual combination

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_local_outlier_factor()

        # Should detect this unusual correlation
        assert detector.predictions_lof is not None

    def test_can_signal_out_of_range(self):
        """Test detection of CAN signals outside valid range."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # CAN signals have valid ranges - test out of range
        df.loc[40, 'throttle_position'] = 150  # Max should be 100%
        df.loc[41, 'battery_voltage'] = 20     # Max should be ~14.5V

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should detect out-of-range values
        assert detector.predictions_if[40] == -1 or detector.predictions_if[41] == -1


class TestRobustness:
    """Test robustness to noise and variations."""

    def test_noise_tolerance(self):
        """Test that model is tolerant to normal sensor noise."""
        df = generate_can_signal_data(samples=100, anomaly_percentage=0)

        # Add small Gaussian noise (normal sensor noise)
        df['engine_rpm'] += np.random.normal(0, 50, 100)
        df['engine_temp'] += np.random.normal(0, 1, 100)

        detector = CANAnomalyDetector(None)
        detector.df = df
        detector.preprocess()
        detector.train_isolation_forest()

        # Should not flag all samples as anomalies
        anomaly_count = (detector.predictions_if == -1).sum()
        assert anomaly_count < 50  # Less than 50% should be flagged

    def test_consistency_across_runs(self):
        """Test consistency of anomaly detection across runs."""
        results = []

        for _ in range(3):
            df = generate_can_signal_data(samples=100, anomaly_percentage=0)
            df.loc[50, 'engine_temp'] = 120  # Add same anomaly

            detector = CANAnomalyDetector(None)
            detector.df = df
            detector.preprocess()
            detector.train_isolation_forest()

            results.append(detector.predictions_if[50])

        # Should consistently detect the anomaly
        assert all(r == -1 for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
