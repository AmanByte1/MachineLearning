# Testing predict_car_price through the Companion app

## 1. Launch the companion with DataLab connected

```powershell
byteflow companion --extension-path D:\MachineLearning\DataLab
```

(adjust the path if your DataLab folder is somewhere else)

Expected: a terminal line before the companion window opens:
```
[extension] loaded: example_hello
[extension] loaded: DataLab
```

If you DON'T see `loaded: DataLab` (or you see `failed: DataLab - ...`),
stop here and paste back that exact line - the feature can't work
in the companion if the extension didn't load, and the error message
will say exactly why (wrong path, missing dependency, etc).

## 2. In the companion chat window, try these in order

**Test A - the exact scenario from before:**
```
predict car price 2028 which runs 10000km
```
Expected: a real predicted price in rupees, the model's accuracy
(R², average error), which details got defaulted (fuel/seller
type/etc), and a warning that 2028 is an extrapolation since it's
outside the training data's year range.

**Test B - a more normal, in-range year:**
```
predict car price for a 2022 model with 30000 km, petrol
```
Expected: same format as above, but WITHOUT the extrapolation
warning, since 2022 is within the range of years the model actually
saw during training.

**Test C - existing dataset predictions (the older feature):**
```
show me price predictions for some cars in the data
```
Expected: a list of several EXISTING cars from car_data.csv with
their actual vs predicted price side by side (this is the
`car_price_predictions` tool - different from Test A/B, which predict
a car that doesn't exist in the data at all).

## 3. If the companion gives a WRONG or hallucinated answer

This is the most important thing to check: the companion's answers
depend on a local LLM (via Ollama) correctly picking the
`predict_car_price` tool for your message - a small/weaker local
model can sometimes still get this wrong even when the tool itself
works perfectly (as proven by `test_predict_car_price_manually.py`
in Test 1/2, which bypass the LLM entirely).

If the companion's answer doesn't look like the expected format above
(e.g. it's inventing numbers, or repeating an old hallucinated error
about a tool that doesn't exist), that tells us the problem is in
tool SELECTION (the LLM's routing decision), not in DataLab's actual
prediction code - which is a different, separate thing to fix than
what we already fixed. Paste back the EXACT question you typed and
the EXACT full reply, so we can tell which case it is.

## 4. Quick sanity check WITHOUT the companion, if step 2 looks wrong

Run this first, before assuming anything is broken in DataLab itself:

```powershell
cd D:\MachineLearning\DataLab
python test_predict_car_price_manually.py
```

If this prints "ALL TESTS PASSED" but the companion still gives a bad
answer, the bug is in tool routing/the LLM, not in DataLab's code -
that narrows down exactly where to look next.
