---
name: code-review
description: Review Python code against an issue, locate the smallest likely defect, and protect unrelated behavior.
---

# Code review workflow

1. Read the issue and relevant implementation before proposing an edit.
2. Compare docstrings, implementation, and tests; identify the smallest contradictory line.
3. Do not edit tests to hide a product defect.
4. Prefer a narrow change and explain why unrelated functions are preserved.
