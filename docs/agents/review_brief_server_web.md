# Code review request (read-only) — RST game server + web client

Review `server/game.py`, `server/bots.py`, `server/app.py`, `server/nlp.py`, `web/assets/app.js` (and `web/index.html` for ids). Context: README.md. Do NOT edit files.

Find **real bugs only** — concurrency/races (asyncio tasks, room.lock, timers vs bot tasks vs submits), turn-order errors, elimination/winner edge cases, reconnect/token flows, host transfer, kicked players, XSS in the client (all server strings rendered via innerHTML must be escaped), WebSocket message validation (malicious client sending odd types/huge payloads), anything that would crash a room or the server, and UI state bugs (stale timers, input enabled when it shouldn't be). 

For each finding: file:line, concrete failure scenario (inputs/sequence → wrong result), severity (high/med/low), and a minimal fix. Rank by severity. Skip style nits. If you're unsure, say so rather than inflating.


Note: the broadcast ordering issue (sequential awaits in app.broadcast letting a slow socket reorder state frames) is already known and being fixed — skip it. Write your findings report as your final message even if partial. TMPDIR=/private/tmp/rst_review if you need temp files.
