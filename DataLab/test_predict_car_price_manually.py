"""
Standalone test for the new predict_car_price feature.

Run this directly to check everything works BEFORE trying it through
ByteFlow's chat interface - it tests the same underlying code, just
without needing a working LLM/planner in the loop, so if something's
wrong you'll know immediately whether it's DataLab's code or
ByteFlow's tool-routing that's the problem.

Usage:
    cd DataLab
    python test_predict_car_price_manually.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("TEST 1: Direct call to DataLab's prediction function")
print("=" * 70)
try:
    from datalab import unit4_regression as reg

    result = reg.predict_price_for_car(year=2028, km_driven=10000)
    print("Result:", result)

    assert result["predicted_price"] > 0, "predicted_price should be a positive number"
    assert result["extrapolating"] is True, "2028 should be flagged as extrapolation"
    print()
    print("PASSED - the underlying prediction function works correctly.")
except Exception as e:
    print()
    print(f"FAILED: {type(e).__name__}: {e}")
    print("This means DataLab's own code has a problem - check that")
    print("pandas/numpy/scikit-learn are installed and datalab/datasets/car_data.csv exists")
    print("(run: python -m datalab.generate_datasets)")
    sys.exit(1)

print()
print("=" * 70)
print("TEST 2: A more normal, in-range prediction (not extrapolating)")
print("=" * 70)
try:
    result = reg.predict_price_for_car(year=2022, km_driven=30000, fuel="Petrol")
    print("Result:", result)
    assert result["extrapolating"] is False, "2022 should be within the training range"
    print()
    print("PASSED")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("TEST 3: The actual ByteFlow tool wrapper (formatted text output)")
print("=" * 70)
try:
    from byteflow_plugin import _predict_car_price

    output = _predict_car_price(2028, 10000)
    print(output)
    assert "Predicted selling price" in output
    assert "EXTRAPOLATION" in output
    print()
    print("PASSED - the ByteFlow tool wrapper works correctly.")
except ModuleNotFoundError as e:
    print(f"SKIPPED: {e}")
    print("This is expected if ByteFlow isn't on your Python path from here.")
    print("This test only matters if run from inside the actual ByteFlow install;")
    print("Tests 1 and 2 above are the ones that matter for DataLab itself.")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("TEST 4: Unknown category is rejected cleanly (not a crash)")
print("=" * 70)
try:
    reg.predict_price_for_car(year=2022, km_driven=30000, fuel="Hydrogen")
    print("FAILED: should have raised an error for an unknown fuel type")
    sys.exit(1)
except ValueError as e:
    print(f"Correctly rejected with a clear message: {e}")
    print()
    print("PASSED")

print()
print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print()
print("If you got here, the new predict_car_price feature is working correctly.")
print("Next, try it for real through ByteFlow:")
print()
print('    byteflow run "predict car price 2028 which runs 10000km" --extension-path <path to DataLab>')
