# Exploratory benchmarks

This directory contains harnesses that are useful for follow-up engineering work but are **not** sources for the reported paper tables.

`web_tool.py` is an earlier single-agent web-tool comparison covering GAIA and SWE-bench Pro. The maintained paper GAIA protocol is `benchmarks.gaia.run`, which adds symmetric tool budgets, explicit infrastructure-error handling, file-task support, per-level reporting, and fairness tests.

Do not publish results from an exploratory harness as a gMAS benchmark result until its protocol, task coverage, and raw artifacts have been reviewed and documented.
