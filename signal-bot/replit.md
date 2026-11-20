# Rules for the agent (read first in every new chat)
- Strategy logic (kernels.py, strategy.py) is validated 1:1 against the backtest. Do NOT change detection logic, parameters, windows or expiries unless explicitly asked.
- Before/after any change: list files with SHA-256 hashes.
- Don't touch secrets. Don't run tests unless asked. Don't touch main branch history.
- Don't ask unnecessary questions. Do only what is asked.
- Every claim in a report must be backed by evidence: commands run, raw output, file:line refs.
- Telegram texts: plain text only, no HTML/Markdown.
