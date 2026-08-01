# LAYER 2 - Terminal (CLI) test

This layer goes through ByteFlow's REAL agent.run() and tool planner
(the actual LLM you have installed via Ollama) - unlike Layer 1, this
CAN fail even if Layer 1 passes, because it depends on your local
model correctly choosing the right tool for what you typed.

If Layer 1 passed but this layer fails: the bug is in tool ROUTING
(the LLM's decision-making), not in DataLab's prediction code.

## Run these one at a time

```powershell
cd D:\ByteFlow

byteflow run "predict car price 2028 which runs 10000km" --extension-path D:\MachineLearning\DataLab

byteflow run "predict the price of a 2022 petrol car with 30000 km" --extension-path D:\MachineLearning\DataLab

byteflow run "give me the raw prediction numbers for a 2028 car with 10000km" --extension-path D:\MachineLearning\DataLab
```

## What to expect

**Command 1 & 2** should call `predict_car_price` and show formatted
text - a rupee amount, the model's R²/MAE, and (only for command 1,
since 2028 is out of range) an extrapolation warning.

**Command 3** is testing the NEW raw/structured tool
(`predict_car_price_raw`) - if your local model is good enough to
notice the phrase "raw prediction numbers" and pick that tool instead
of the formatted one, you'll see a Python dict printed instead of
nice sentences. It's fine if it doesn't - this is testing a smaller,
newer capability, not a core one.

## Diagnosing a wrong answer at this layer

Since this uses your REAL local model, a wrong or hallucinated answer
here (while Layer 1 passed) tells you specifically: your model failed
to choose the correct tool for this phrasing, or answered from plain
chat instead of calling the tool at all. That's a model/prompt problem,
not a DataLab code problem - rephrasing the request more explicitly
("use the predict_car_price tool for a 2028 car with 10000km") is a
reasonable thing to try, and if that STILL fails, it may point to your
local model being too weak for reliable tool selection with this many
tools registered - a real, known limitation of small local models
that isn't something DataLab's code can fix.
