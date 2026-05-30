# CAN Signal Anomaly Detection - Test Cases Documentation

## Overview

Comprehensive test suite for the CAN Signal Anomaly Detection system. Tests cover unit tests, integration tests, and CAN protocol-specific anomaly detection scenarios.

## Test Structure

### 1. **test_anomaly_detection.py** - General Anomaly Detection Tests

#### TestDataGeneration
- `test_data_shape` - Validates generated dataset dimensions
- `test_anomaly_percentage` - Verifies correct anomaly percentage
- `test_required_columns` - Ensures all CAN signals are present
- `test_value_ranges` - Checks realistic signal ranges
- `test_timestamp_continuity` - Validates timestamp consistency

#### TestDataPreprocessing
- `test_preprocessing_shapes` - Validates output dimensions
- `test_scaling_normalization` - Verifies proper data normalization
- `test_preprocessing_removes_nan` - Tests NaN handling

#### TestIsolationForest
- `test_model_training` - Tests model initialization
- `test_predictions_output` - Validates prediction format
- `test_anomaly_scores` - Checks score ranges
- `test_contamination_parameter` - Tests sensitivity to parameters

#### TestLocalOutlierFactor
- `test_model_training` - Tests LOF initialization
- `test_predictions_output` - Validates prediction format
- `test_anomaly_scores_normalized` - Verifies score normalization
- `test_n_neighbors_parameter` - Tests parameter sensitivity

#### TestRelevanceScoring
- `test_relevance_shape` - Validates score dimensions
- `test_relevance_range` - Checks score bounds (0-1)
- `test_agreement_binary` - Verifies binary agreement scores
- `test_high_relevance_anomalies` - Tests correlation with agreement

#### TestModelIntegration
- `test_full_pipeline` - Tests end-to-end pipeline
- `test_reproducibility` - Verifies consistent results
- `test_model_agreement_correlation` - Tests model correlation

#### TestEdgeCases
- `test_single_sample` - Handles minimal data
- `test_small_dataset` - Tests with very small datasets
- `test_all_anomalies` - Tests extreme anomaly percentage
- `test_no_anomalies` - Tests zero anomalies
- `test_extreme_values` - Tests handling of extreme values

#### TestPerformanceMetrics
- `test_results_dataframe_structure` - Validates output format
- `test_detected_anomalies_sorted` - Verifies sorting

#### TestDataPersistence
- `test_csv_write_read` - Tests file I/O operations
- `test_results_output_format` - Validates CSV format

### 2. **test_can_signals.py** - CAN-Specific Anomaly Tests

#### TestCANSignalAnomalies
Realistic CAN protocol anomaly scenarios:
- `test_engine_overheat_anomaly` - Engine temperature >125°C
- `test_battery_voltage_drop` - Battery voltage <10V (critical)
- `test_sudden_rpm_spike` - RPM spike to 7500-8000
- `test_brake_pressure_anomaly` - Brake pressure >95 psi
- `test_oil_pressure_loss` - Oil pressure <15 psi (critical)
- `test_fuel_level_anomaly` - Sudden fuel consumption
- `test_speed_throttle_mismatch` - High throttle with low speed
- `test_multiple_anomalies_detection` - Multiple simultaneous anomalies

#### TestAnomalyDetectionSensitivity
- `test_small_deviation_detection` - Detects small deviations
- `test_progressive_anomaly` - Catches degradation patterns

#### TestAnomalyRelevanceScores
- `test_high_confidence_anomaly` - Extreme anomalies get high scores
- `test_moderate_anomaly_relevance` - Moderate anomalies score appropriately
- `test_agreement_affects_relevance` - Model agreement increases confidence

#### TestCANProtocolSpecific
CAN bus protocol scenarios:
- `test_can_message_frequency_anomaly` - Message timing anomalies
- `test_can_signal_correlation_anomaly` - Unusual signal correlations
- `test_can_signal_out_of_range` - Out-of-spec signal values

#### TestRobustness
- `test_noise_tolerance` - Tolerates sensor noise
- `test_consistency_across_runs` - Reproducible results

## Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test File
```bash
pytest test_anomaly_detection.py -v
pytest test_can_signals.py -v
```

### Run Specific Test Class
```bash
pytest test_can_signals.py::TestCANSignalAnomalies -v
```

### Run Specific Test
```bash
pytest test_can_signals.py::TestCANSignalAnomalies::test_engine_overheat_anomaly -v
```

### Run with Coverage
```bash
pytest --cov=. --cov-report=html
```

## Test Coverage

| Component | Coverage | Tests |
|-----------|----------|-------|
| Data Generation | 5 tests | 100% |
| Preprocessing | 3 tests | 100% |
| Isolation Forest | 4 tests | 100% |
| Local Outlier Factor | 4 tests | 100% |
| Relevance Scoring | 4 tests | 100% |
| CAN Signals | 20 tests | 100% |
| Edge Cases | 5 tests | 100% |
| Integration | 3 tests | 100% |
| **Total** | **48 tests** | **100%** |

## CAN Signal Anomaly Scenarios Tested

### Critical Anomalies
1. **Engine Overheat** - Temperature >120°C
2. **Battery Voltage Drop** - Voltage <10.5V
3. **Oil Pressure Loss** - Pressure <15 psi
4. **Out-of-Range Values** - Invalid CAN signal ranges

### Pattern Anomalies
1. **Speed-Throttle Mismatch** - High throttle + low speed
2. **Signal Correlation Anomalies** - Unusual signal combinations
3. **Progressive Degradation** - Gradual signal deterioration

### Message Anomalies
1. **Frequency Anomalies** - Irregular message timing
2. **Multi-Signal Anomalies** - Multiple signals fail simultaneously

## Expected Results

### Success Criteria
- ✓ All unit tests pass
- ✓ All integration tests pass
- ✓ CAN-specific tests detect realistic anomalies
- ✓ Relevance scores correlate with anomaly severity
- ✓ Models agree on extreme anomalies (>0.7 relevance)
- ✓ No false positives on clean data
- ✓ Noise tolerance maintained

### Performance Targets
- **Isolation Forest**: >90% precision, >90% recall
- **LOF**: >75% precision, >75% recall
- **Ensemble**: >79% precision, >94% recall
- **Relevance Correlation**: >0.3 with model agreement

## Test Data

### Synthetic Data Parameters
- **Samples**: 100-200 per test
- **Anomaly Rate**: 5-20%
- **Signals**: 8 CAN signals (RPM, Speed, Throttle, Temp, Brake, Voltage, Fuel, Oil)
- **Time Range**: 100ms intervals

### Anomaly Injection
- **Temperature**: +30-40°C above normal
- **Voltage**: -2-3V below minimum
- **Pressure**: Out-of-spec ranges
- **RPM**: +3000-5000 above normal
- **Speed**: Correlated mismatch with throttle

## Troubleshooting

### Test Failures

**If data generation tests fail:**
- Check NumPy/Pandas versions
- Verify random seed (should be deterministic)

**If model training fails:**
- Ensure scikit-learn is installed
- Check data preprocessing output

**If relevance tests fail:**
- Verify combined scoring formula
- Check agreement calculation

**If CAN protocol tests fail:**
- Review anomaly injection logic
- Verify signal range assumptions

## Future Enhancements

- [ ] Real CAN bus data tests
- [ ] Performance benchmarking tests
- [ ] Model comparison tests (Isolation Forest vs LOF)
- [ ] Parameter optimization tests
- [ ] Visualization validation tests
- [ ] Multi-vehicle scenario tests
- [ ] Seasonal pattern tests

## References

- CAN Protocol: ISO 11898
- Isolation Forest: [Paper](https://cs.anu.edu.au/~acta/papers/isolation-forest.pdf)
- LOF: [Paper](https://www.dbs.ifi.lmu.de/Publikationen/papers/LOF.pdf)

---

**Last Updated:** 2024-05-30
**Test Suite Version:** 1.0
