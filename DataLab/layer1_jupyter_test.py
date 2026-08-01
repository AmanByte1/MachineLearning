"""
LAYER 1 - Pure code test (works the same in a script OR pasted cell by
cell into Jupyter). No LLM, no agent, no companion - this tests
DataLab's actual prediction code directly, the same way you'd use it
in a notebook.

If this layer FAILS: the bug is in DataLab's own code (unit4_regression.py
or byteflow_plugin.py) - fix it there.

If this layer PASSES but Layer 2 (terminal) or Layer 3 (companion)
fails: the bug is NOT in the prediction code itself - it's in how
ByteFlow's LLM is routing/selecting tools, which is a completely
different, separate thing to fix.

Run as a script:
    python layer1_jupyter_test.py

Or paste the numbered cells below one at a time into a Jupyter notebook.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Also need ByteFlow importable (byteflow_plugin.py imports from it) -
# adjust this path if your ByteFlow folder is somewhere else.
_BYTEFLOW_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ByteFlow"),
    r"D:\ByteFlow",
    r"D:\MachineLearning\ByteFlow",
]
for candidate in _BYTEFLOW_CANDIDATES:
    if os.path.isdir(os.path.join(candidate, "byteflow")):
        sys.path.insert(0, candidate)
        break

# ============================================================
# CELL 1 - direct call to the underlying regression function
# ============================================================
from datalab import unit4_regression as reg

result = reg.predict_price_for_car(year=2028, km_driven=10000)
print("Cell 1 - raw regression function:")
print(result)
assert result["predicted_price"] > 0
assert result["extrapolating"] is True
print("Cell 1 PASSED\n")

# ============================================================
# CELL 2 - the structured tool function (what ByteFlow actually calls)
# ============================================================
from byteflow_plugin import _predict_car_price_structured, format_car_price_prediction, _predict_car_price

structured = _predict_car_price_structured(2028, 10000)
print("Cell 2 - structured tool output:")
print(structured)
assert "predicted_price" in structured
print("Cell 2 PASSED\n")

# ============================================================
# CELL 3 - the formatted text version (what a human actually reads)
# ============================================================
formatted = _predict_car_price(2028, 10000)
print("Cell 3 - formatted text:")
print(formatted)
assert "Predicted selling price" in formatted
assert "EXTRAPOLATION" in formatted
print("Cell 3 PASSED\n")

# ============================================================
# CELL 4 - an in-range prediction (no extrapolation warning expected)
# ============================================================
in_range = _predict_car_price(2022, 30000, fuel="Petrol")
print("Cell 4 - in-range prediction:")
print(in_range)
assert "EXTRAPOLATION" not in in_range
print("Cell 4 PASSED\n")

# ============================================================
# CELL 5 - bad input is rejected cleanly, not a crash
# ============================================================
bad_input = _predict_car_price(2022, 30000, fuel="Hydrogen")
print("Cell 5 - unknown category:")
print(bad_input)
assert "[Error:" in bad_input
print("Cell 5 PASSED\n")

print("=" * 60)
print("ALL LAYER 1 (pure code) TESTS PASSED")
print("If ByteFlow's terminal or companion give a wrong answer despite")
print("this passing, the problem is in tool ROUTING, not in this code.")
print("=" * 60)
