# LAYER 3 - Companion (GUI) test

Same underlying code as Layer 2, but through the companion window
instead of one-off terminal commands. Companion and terminal use the
SAME agent.run() and the SAME tool planner, so in principle they
should behave the same - if one works and the other doesn't, that
itself is useful information (see "if companion misbehaves but
terminal doesn't" below).

## Launch

```powershell
byteflow companion --extension-path D:\MachineLearning\DataLab
```

Confirm you see this in the terminal BEFORE the window opens:
```
[extension] loaded: example_hello
[extension] loaded: DataLab
```

## Try these in the chat window, in order

1. `predict car price 2028 which runs 10000km`
2. `predict the price of a 2022 petrol car with 30000 km`
3. `show me price predictions for some cars in the data`

Expected results are the same as Layer 2's - see layer2_terminal_test.md.

## How to use the 3 layers together to isolate a bug

Run all three layers in order (1 = pure code, 2 = terminal, 3 =
companion) and note where it FIRST breaks:

| Layer 1 | Layer 2 | Layer 3 | What that means |
|---|---|---|---|
| PASS | PASS | PASS | Everything genuinely works |
| FAIL | - | - | Bug in DataLab's own code - fix unit4_regression.py / byteflow_plugin.py |
| PASS | FAIL | FAIL | Bug in tool routing/LLM - same for both surfaces, likely your model or a planner prompt issue |
| PASS | PASS | FAIL | Bug SPECIFIC to companion - the terminal proves the code and routing both work, so look at companion.py specifically (state handling, message queueing, or how it constructs/uses the agent) rather than touching DataLab or the general routing logic at all |
| PASS | FAIL | PASS | Unusual - would suggest something timing/session-specific to the terminal invocation; re-run Layer 2 to rule out a one-off model hiccup before assuming this pattern is real |

That last "PASS / PASS / FAIL" row is exactly the case you asked
about - if that's what you get, tell me specifically that pattern and
paste the exact companion transcript, and the fix should be scoped to
companion.py only, not the shared agent/tool code (since Layer 2
proves that part is fine).
