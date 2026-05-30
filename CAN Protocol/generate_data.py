import pandas as pd
import numpy as np
import os

def generate_can_signal_data(samples=1000, anomaly_percentage=5):
    """
    Generate synthetic CAN signal data with realistic patterns and anomalies.

    Args:
        samples: Number of samples to generate
        anomaly_percentage: Percentage of anomalous samples

    Returns:
        DataFrame with CAN signals and anomaly labels
    """
    np.random.seed(42)

    # Generate timestamps
    timestamps = pd.date_range('2024-01-01', periods=samples, freq='100ms')

    # Generate normal CAN signals
    data = {
        'timestamp': timestamps,
        'engine_rpm': np.random.normal(3000, 500, samples),  # 2500-3500 RPM normal
        'vehicle_speed': np.random.normal(60, 15, samples),  # 45-75 km/h normal
        'throttle_position': np.random.normal(40, 20, samples),  # 0-100%
        'engine_temp': np.random.normal(90, 5, samples),  # 85-95°C normal
        'brake_pressure': np.random.normal(20, 10, samples),  # 0-100 psi
        'battery_voltage': np.random.normal(13.5, 0.3, samples),  # 13-14V normal
        'fuel_level': np.linspace(100, 20, samples) + np.random.normal(0, 2, samples),  # Decreasing
        'oil_pressure': np.random.normal(50, 8, samples),  # 40-60 psi normal
    }

    df = pd.DataFrame(data)

    # Add anomalies
    num_anomalies = int(samples * anomaly_percentage / 100)
    anomaly_indices = np.random.choice(samples, num_anomalies, replace=False)

    df['is_anomaly'] = 0

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['high', 'low', 'spike', 'pattern'])

        if anomaly_type == 'high':
            df.loc[idx, 'engine_rpm'] = np.random.uniform(6000, 8000)
            df.loc[idx, 'engine_temp'] = np.random.uniform(110, 130)
        elif anomaly_type == 'low':
            df.loc[idx, 'battery_voltage'] = np.random.uniform(10, 11)
            df.loc[idx, 'oil_pressure'] = np.random.uniform(15, 25)
        elif anomaly_type == 'spike':
            df.loc[idx, 'throttle_position'] = np.random.uniform(90, 100)
            df.loc[idx, 'engine_rpm'] = np.random.uniform(7000, 9000)
        else:  # pattern
            df.loc[idx, 'brake_pressure'] = np.random.uniform(80, 100)
            df.loc[idx, 'vehicle_speed'] = np.random.uniform(5, 10)

        df.loc[idx, 'is_anomaly'] = 1

    return df

if __name__ == '__main__':
    # Generate data
    df = generate_can_signal_data(samples=1000, anomaly_percentage=5)

    # Save to CSV
    output_path = 'can_signal_data.csv'
    df.to_csv(output_path, index=False)

    print(f"[OK] Generated {len(df)} CAN signal samples")
    print(f"[OK] Anomalies: {df['is_anomaly'].sum()}")
    print(f"[OK] Saved to: {output_path}")
    print(f"\nDataset preview:")
    print(df.head(10))
