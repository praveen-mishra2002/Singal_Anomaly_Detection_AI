#!/usr/bin/env python
"""
Test runner for CAN Signal Anomaly Detection System
Runs all test suites and generates a comprehensive report
"""

import subprocess
import sys
import os
from datetime import datetime


def run_tests():
    """Run all tests and generate report."""
    print("=" * 80)
    print("CAN SIGNAL ANOMALY DETECTION - TEST SUITE")
    print("=" * 80)
    print(f"\nTest Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    # Test files to run
    test_files = [
        'test_anomaly_detection.py',
        'test_can_signals.py'
    ]

    all_passed = True
    results = {}

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"[SKIP] {test_file} - File not found")
            continue

        print(f"\n[RUNNING] {test_file}")
        print("-" * 80)

        cmd = [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short']

        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        results[test_file] = result.returncode == 0
        if result.returncode != 0:
            all_passed = False

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_file, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{status} {test_file}")

    print("-" * 80)
    print(f"Test End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
