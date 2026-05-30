"""
Pytest configuration and import setup for CAN anomaly detection tests.
Handles imports from multiple directory structures.
"""

import sys
import os
from pathlib import Path

# Get the directory containing this file
test_dir = Path(__file__).parent.resolve()
code_dir = test_dir.parent / 'Code'
root_dir = test_dir.parent.parent.parent

# Add directories to Python path in order of preference
paths_to_add = [
    str(code_dir),  # CAN Protocol/Code
    str(test_dir),  # CAN Protocol/Test Cases
    str(root_dir),  # Root directory
    str(root_dir / 'CAN Protocol' / 'Code'),
    str(root_dir / 'CAN Protocol' / 'Data'),
]

for path in paths_to_add:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)

# Try to import modules with proper error handling
def safe_import(module_name, from_paths=None):
    """Safely import a module from multiple possible locations."""
    try:
        return __import__(module_name)
    except ImportError as e:
        if from_paths:
            for path in from_paths:
                if os.path.exists(os.path.join(path, f'{module_name}.py')):
                    sys.path.insert(0, path)
                    try:
                        return __import__(module_name)
                    except ImportError:
                        continue
        raise ImportError(f"Could not import {module_name}") from e

# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with proper paths."""
    # Ensure Code directory is in path
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    # Verify modules can be imported
    try:
        import detect_anomalies
        import generate_data
    except ImportError as e:
        print(f"Warning: Import configuration incomplete: {e}")

# Test markers
def pytest_configure_markers(config):
    """Define custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "canprotocol: mark test as CAN protocol specific"
    )
