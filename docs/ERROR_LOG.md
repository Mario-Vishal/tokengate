# ContextPilot — ERROR LOG

Record every meaningful error: what happened, root cause, fix, and prevention.
Newest on top.

Format:
```
### YYYY-MM-DD — short title
- Context: where/when it happened
- Symptom: the actual error/message
- Root cause:
- Fix:
- Prevention / follow-up:
```

---

_No errors logged yet (Phase 0 — planning)._

## Watch-list (anticipated risks)
- **Python 3.14 wheels:** optional `tiktoken` extra may lack a prebuilt wheel for
  3.14 on Windows. Mitigation: core has no required deps; tiktoken is opt-in.
- **Token-count drift:** heuristic counter under/over-estimates vs real tokenizers,
  risking budget overshoot. Mitigation: budgeter uses a safety margin and the audit
  reports real counts; document the approximation.
