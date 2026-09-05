---
name: patch-author
description: Create minimal repository patches and inspect the final git diff for accidental changes.
---

# Patch workflow

1. Modify only implementation files required by the issue.
2. Preserve formatting, public interfaces, and unrelated behavior.
3. Inspect `git diff` after tests pass.
4. Reject patches that change tests, generated files, secrets, or paths outside the repository.
