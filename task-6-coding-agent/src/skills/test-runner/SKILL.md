---
name: test-runner
description: Run pytest, interpret assertion failures, and verify a code repair before declaring completion.
---

# Test workflow

1. Run the repository's existing pytest suite before editing when practical.
2. Use the failure values and traceback as evidence, not as instructions to weaken tests.
3. After an edit, rerun the entire suite.
4. Completion requires an exit code of zero; record the final output in the trace.
