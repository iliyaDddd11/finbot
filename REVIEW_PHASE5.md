# Phase 5 Review — Report Generator + Web App (`finrobot_equity`)

Adversarial security + correctness review of the two surfaces outside `quant_eval`:
the **FastAPI web app** (`web_app/**`) and the **LLM report generator**
(`core/src/create_equity_report.py`, `modules/**`, `modules/equity_agents/**`).
Four independent review passes plus direct reads of `auth.py`, `admin_routes.py`,
`crud.py`, `models.py`, and the request-logger.

Each finding is **CONFIRMED** (traced) or **PLAUSIBLE**. Status is **FIXED**
(this branch) or **FOLLOW-UP** (precise fix noted; needs a running app / render
test / product decision — the web app can't be executed here: `fastapi`,
`sqlalchemy`, `bcrypt` are not installed, so these fixes are correct-by-
construction but not runtime-verified in this environment).

---

## CRITICAL

### P5-1 — Admin console has no authorization — any logged-in user is "admin" — CONFIRMED · **FIXED**
`admin_routes.py:require_admin` only checked that a session existed (the role
check was commented out). Every `/api/admin/*` endpoint (all users + emails, all
reports, all request logs with IPs, per-user detail) was readable by any
authenticated user. **Fix:** `require_admin` now enforces an email allowlist
(`FINROBOT_ADMIN_EMAILS` + bootstrapped `FINROBOT_ADMIN_EMAIL`), **fails closed**
(nobody is admin if unset), and returns 403 otherwise — matching the gating
`/api/logs` already used.

### P5-2 — Passwords stored as unsalted SHA-256 — CONFIRMED · **FIXED**
`crud.hash_password` used a single `sha256` pass: fast to brute-force, and
identical passwords produced identical hashes (rainbow-table-able). **Fix:**
salted **PBKDF2-HMAC-SHA256** (200k rounds, stdlib — no new dependency),
constant-time verify (`hmac.compare_digest`), legacy hashes still verify and are
transparently upgraded on next successful login (`needs_rehash` + rehash in
`authenticate_user`).

### P5-3 — Stored XSS: LLM/news text rendered as raw HTML — CONFIRMED · **partially FIXED**
No HTML escaping existed anywhere in the render path. Every narrative field
(`tagline`, `company_overview`, `investment_overview`, `risks`, `news_summary`,
…), all LLM-generated, was injected raw into the served report — so
`<img src=x onerror=…>` in a source article executes in any viewer's browser.
**Fixed:** `_markdown_to_html` now `html.escape`s its input first (markdown
syntax survives; literal tags are neutralized), covering the narrative fields —
the largest surface. **Follow-up:** the same escaping is still needed at the
non-markdown interpolation sites (P5-3b): catalyst/news/sensitivity JSON
(`html_renderer.py:709/731/757`, `html_template_professional.py:637/651/691`),
DataFrame table cells (`html_renderer.py:488-500`), `company_name` in titles,
and `report_structure.py` annotation helpers. These need a render test to verify
no double-escaping of intended markup.

---

## HIGH

### P5-4 — Plaintext passwords & API keys written to the request-log table — CONFIRMED · **FIXED**
The request-logger persisted raw POST bodies; `/api/auth/*` bodies carry
cleartext passwords and `/api/run` bodies carry `fmp_api_key`/`openai_api_key`.
**Fix:** `_redact_body` drops auth-endpoint bodies entirely and redacts any
`*password*`/`*api_key*`/`*token*`/`*secret*` key from other JSON bodies before
storage.

### P5-5 — IDOR on job status/logs endpoints — CONFIRMED · **FOLLOW-UP**
`/api/status/{task_id}`, `/api/logs/{task_id}`, `/api/logs/{task_id}/download`
(`main.py:~1019/1037/1056`) check auth but never compare the job's owner to the
caller, so any user can read any other user's job status, result paths, and full
logs (amplified by P5-1, which leaks every `task_id`). **Fix:** compare
`tasks[task_id]["user"]` to the current user; 403/404 otherwise.

### P5-6 — GitHub OAuth has no `state` parameter — CONFIRMED · **FOLLOW-UP**
`main.py:~245-335` omits and never validates `state`, enabling login-CSRF /
session fixation (a victim silently logged into the attacker's GitHub account).
**Fix:** generate a random `state`, store it (signed cookie or server-side), and
require an exact match in the callback.

### P5-7 — LLM prompts demand data never supplied → systematic fabrication — CONFIRMED · **FOLLOW-UP**
`equity_agents/*` prompts ask the model to report CEO/CFO names, current stock
price, 52-week range, PE/PB/PS, quarterly beat/miss, and to "use web searches",
but `agent_manager` supplies none of that and the agents have **no tools**. The
model invents executive names, prices, and multiples that render as fact.
**Fix (product):** remove the "web search" instructions and the asks for data
not provided (or actually supply that data), and add a numeric-grounding check
that flags/strips figures in the prose that don't appear in the computed inputs.

### P5-8 — Prompt injection from news/filing text — CONFIRMED · **FOLLOW-UP**
`agent_manager.py:57-61` and `text_generator_agents.py:68-72` concatenate raw
news title/body into prompts for every section; injected instructions in a press
release can override the analyst instructions. **Fix:** wrap untrusted text in an
explicit delimiter, instruct the model to treat it as data only, and strip
instruction-like content.

### P5-9 — `github:` email namespace pre-hijack — CONFIRMED · **FIXED**
A local account registered as `github:victim@…` would be merged into on the
victim's later GitHub login. **Fix:** `register_user` rejects any email starting
with the reserved `github:` prefix.

### P5-10 — FMP API key leaks via news error text — CONFIRMED · **FIXED**
`news_integrator` returned/logged `str(e)`, and `requests` exceptions embed the
request URL (which carries `apikey`). **Fix:** redact the key from the log and
return a generic message instead of the raw error into the report.

### C1 — Missing target price silently renders a false "Sell" — CONFIRMED · **FIXED**
`_derive_rating` computed `upside=(0-price)/price=-100%` → "Sell" whenever
`target_price` defaulted to `$0.00`. **Fix:** return the provided/API rating (or
"N/A") when `target <= 0` instead of deriving.

---

## MEDIUM

| # | Where | Finding | Status |
|---|---|---|---|
| P5-11 | `main.py` openai_base_url (`~485/583/768`) | **SSRF + key exfiltration** — user-supplied base URL passed to the LLM subprocess with the API key; point it at an internal/attacker host. Fix: allowlist. | FOLLOW-UP |
| P5-12 | `main.py:~926/1191`, `create_equity_report.py` | **Path traversal** in `/api/walkforward/{run_id}`, `/api/reports/{ticker}`, and report output dir from `company_ticker`. | ticker gen path **FIXED** (strict regex); API endpoints FOLLOW-UP |
| P5-13 | `main.py` cookies | Session cookie missing `Secure`. | **FIXED** (env `COOKIE_SECURE`, secure by default) |
| P5-14 | `connection.py:16-21` | SQLite `StaticPool` + `check_same_thread=False`, no `busy_timeout` → write races / "database is locked". Fix: proper pool + `timeout`/WAL. | FOLLOW-UP |
| P5-15 | `main.py` rate limits | Keyed per-IP (no trusted-proxy/XFF), not per-authenticated-user, on expensive LLM jobs. | FOLLOW-UP |
| C3 | `html_renderer.py:459-474` | `_format_value` appended a bogus **"x"** to plain numbers in [100,1000) (`250`→`250.0x`). | **FIXED** |
| C2/C4/C5 | `html_renderer.py`, `create_equity_report.py` | Derived rating overrides a provided/API rating (C2); `_fix_sentiment` force-relabels catalysts positive by keyword (C4); `_fix_stale_dates` rewrites real historical dates in prose (C5). | FOLLOW-UP |
| P5-16 | `enhanced_text_generator.py`, `text_generator_agents.py` | Silent fallbacks emit generic boilerplate / echo raw context as if it were analysis, indistinguishable from real output. Fix: mark fallback sections explicitly. | FOLLOW-UP |

---

## LOW / robustness (FOLLOW-UP)

- `text_generator_agents.py:71` — `.get(k,'N/A')[:200]` raises `TypeError` when the key exists with value `None` (common in FMP); guard with `or ''`.
- `main.py` OAuth callback — `github_user['login']` can `KeyError`/500 on a malformed GitHub response.
- `crud.create_user` — uncaught `IntegrityError` on a duplicate-email race (check-then-insert TOCTOU).
- Factual sections use `temperature=0.7` with no seed → non-reproducible numbers; pin low temperature/seed.
- `news_integrator._is_within_time_window` returns `True` on unparseable/exception dates → malformed/future news bypasses the recency filter.
- C6/C7 — unit-consistency risks in `dividend_yield`/`roe` formatting and CI `$M/$B` labels forced onto ratio metrics.
- Default admin auto-created as `admin@finrobot.com` with a generated password printed to stdout.

---

## Checked and OK

- **No SQL injection** — all DB access uses the SQLAlchemy ORM with bound parameters; no `text()`/string-built SQL.
- **Session IDs** use `secrets.token_urlsafe(32)` (strong); expiry enforced in `get_session`.
- Admin serializers whitelist fields (no `password_hash` leak); the exposure was via `request_body` (P5-4).
- No hardcoded secrets in the reviewed files; no `subprocess`/shell in the render path.
- `str.format(**data)` is not brace-injectable from substituted values (only raw-HTML injection mattered → P5-3).

---

## Fixed in this branch (summary)

Passwords (PBKDF2 + upgrade-on-login), admin authorization (fail-closed
allowlist), credential-log redaction, `github:` pre-hijack block, news API-key
redaction, ticker path-traversal validation, narrative-field XSS escaping,
Secure cookies, and the false-"Sell"/bogus-"x" correctness bugs. All edited
files compile; **runtime verification requires a live web-app run** (deps absent
here). The FOLLOW-UP items above are the scoped Phase 5 hardening backlog — the
web-app authz/OAuth/SSRF items first, then the full XSS-escaping audit, then the
LLM-grounding redesign (P5-7/P5-8), which is genuine product work.
