# FraudLens security notes

What the MVP protects today, and what still has to be added before it touches real card data.

## In place

| Area | Control |
|---|---|
| Authentication (app) | Username + password. PBKDF2-HMAC-SHA256, 600,000 iterations, random 16-byte salt per user. Constant-time comparison. Same error and same timing for unknown users and wrong passwords (no user enumeration). |
| Brute force | Account locked for `app.lockout_minutes` after `app.max_failed_logins` wrong passwords. Failed logins and lockouts are written to the admin audit log. |
| Sessions | Idle timeout (`app.session_timeout_minutes`). Role and active status are re-read from the database on every interaction, so disabling a user or changing a role takes effect immediately. |
| Password policy | At least 10 characters, a letter and a digit, must not contain the username. Admin-set passwords are temporary: the user must choose a new one at first sign-in. |
| Authorisation | Three roles (viewer, analyst, admin) and one permission table (`fraudkit/auth.py: PERMISSIONS`). Pages and buttons check permissions server-side; hidden buttons are not the only guard. The last active admin cannot be demoted or disabled. |
| API keys | 32-byte random tokens (`fl_...`), shown once, stored only as SHA-256 hashes. Revocable, with last-used time and request count. Optional single bootstrap key via `FRAUDLENS_API_KEY`, compared in constant time. |
| API hardening | Per-key rate limit (`api.rate_limit_per_minute`, 429 + `Retry-After`), batch size limit (413), strict input schema with bounds (422), request IDs, no stack traces in responses, `nosniff` and `no-store` headers. |
| Audit | Case decisions (who, what, before/after status, note, time) and admin/security events (logins, failed logins, user and role changes, key creation/revocation, retraining, queue deletion), exportable as CSV. |
| Data minimisation | The case queue stores only the last four digits of a card number. API logs never contain card numbers. |
| Output encoding | All data-derived text shown with HTML formatting is escaped (fixes a stored-XSS path from uploaded CSV merchant names into the case queue). |
| CSV injection | Every CSV download prefixes cells starting with `=`, `+`, `-`, `@`, tab or CR with `'`. |
| Uploads | 50 MB cap (Streamlit), row cap (`app.max_upload_rows`), schema validation with readable errors. |
| Artifacts | Training writes every artifact to a temp file and swaps it in atomically, so the app and API never load a half-written model. Each model has a version id (`<UTC time>-<sha256 prefix>`) that is returned by the API and stored on every case. |
| Network | The app binds to `localhost` by default. Docker Compose publishes ports on `127.0.0.1` only. XSRF protection is on. The container runs as a non-root user. |
| Secrets | `.env`, `kaggle.json`, databases, models and data are git-ignored. `.env.example` documents the variables without values. |

## Required before production with real card data

- **Single sign-on** (OIDC/SAML) and MFA instead of local passwords.
- **TLS** termination in front of both services (reverse proxy or load balancer). Never expose the ports directly.
- **A server database** (PostgreSQL) instead of SQLite, with backups and encryption at rest.
- **A shared rate limiter** (gateway or Redis). The in-process limiter is per worker.
- **A secrets manager** for API keys and DB credentials.
- **PCI DSS scope review.** The scoring path receives full card numbers (needed for per-card history). Tokenise them upstream, or keep the scoring service inside the cardholder-data environment.
- **Model governance**: approval step before a retrained model is served, plus monitoring alerts wired to on-call.
- **Pickle trust boundary**: `model.joblib` is a pickle. Only load models produced by your own training pipeline, from storage that only it can write.

## Reporting

This is a demo/MVP repository. Please report security issues privately to the repository owner rather than in a public issue.
