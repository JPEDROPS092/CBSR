# Security and privacy

The platform handles medical images. The design assumption is that any of them
could be traced back to a person, so identifiers, access and logs are treated
accordingly.

## Authentication

* Passwords are stored as PBKDF2-HMAC-SHA256 with a 16-byte random salt and
  600 000 iterations by default, in the form
  `pbkdf2_sha256$<iterations>$<salt>$<hash>`. The cost is configurable and old
  hashes are transparently upgraded on the next successful login.
* Verification is constant-time (`hmac.compare_digest`). A login for an unknown
  email still performs a hash, so response timing does not reveal whether an
  account exists — and the error message is identical for unknown email, wrong
  password and deactivated account.
* Tokens are JWTs carrying `sub`, `role`, `jti`, `iat`, `exp` and `typ`. A
  refresh token is rejected wherever an access token is required. Access tokens
  default to 30 minutes, refresh tokens to 14 days.
* Password policy: minimum 12 characters, at least one letter and one digit
  (`PASSWORD_MIN_LENGTH`).

## Authorization

Five roles, enforced per route by an explicit permission:

| Permission | admin | doctor | researcher | operator | viewer |
|---|:-:|:-:|:-:|:-:|:-:|
| `patient:read` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `patient:write` | ✓ | ✓ | | ✓ | |
| `exam:read` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `exam:write` | ✓ | ✓ | | ✓ | |
| `image:upload` | ✓ | ✓ | | ✓ | |
| `analysis:read` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `analysis:run` | ✓ | ✓ | ✓ | | |
| `report:read` | ✓ | ✓ | ✓ | | ✓ |
| `report:write` | ✓ | ✓ | | | |
| `model:read` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `model:manage`, `user:manage`, `audit:read` | ✓ | | | | |

Researchers work on cohorts: they may run models and read results but not
create or edit patient records. Operators acquire and upload but do not run or
interpret analyses.

## What is deliberately not stored

* **No patient names.** The `patients` table has `external_ref` (a site-assigned
  pseudonym), `birth_year`, `sex`, a consent flag and non-identifying clinical
  context. Nothing else.
* **No national identifiers.** An `external_ref` shaped like a CPF is rejected
  with 422.
* **No full dates of birth** — the year only, which is what age-stratified
  analysis needs.
* **No uploaded filenames.** They routinely contain patient names; only the
  extension is kept.
* **No identifiers in URLs or storage keys.** Every path is built from UUIDs.

## Uploads

* Type is decided by magic number, never by the client's `Content-Type` or the
  file extension.
* Size is capped by `MAX_UPLOAD_BYTES` (100 MB default), checked before the
  bytes are decoded.
* Every upload is decoded before being accepted, so a file that is not a
  readable image never reaches a model.
* Decompression-bomb protection is Pillow's, raised to a ceiling that still
  covers wide-field fundus and OCT volumes.
* Storage keys are validated against path traversal.
* DICOM is currently refused: accepting it means de-identifying the header
  first, and doing that half-way is worse than not doing it.

## Object storage

Pixel data never enters PostgreSQL. Masks, overlays and rendered reports are
derived medical data and are never served from a public bucket: with
S3-compatible storage the API issues short-lived presigned URLs
(`PRESIGNED_URL_TTL_SECONDS`, 15 min default); with local storage the bytes are
streamed through `/api/v1/objects/{key}`, which enforces the same
authentication and RBAC as every other route. `STORAGE_BACKEND=local` is
refused outside `local`/`test` environments.

## Logging and errors

* Logs are structured JSON with correlation ids (request, analysis, model,
  user). They contain identifiers, never image content, patient attributes or
  request bodies.
* Client IPs are stored in the audit trail only as a salted hash — enough to
  correlate abuse, not enough to re-identify a person from the log alone.
* Unexpected exceptions are logged with a stack trace and answered with a
  generic message. Internal details never reach an API client; in a medical
  system an error string can carry patient data.
* Validation errors name the field and the reason but do not echo the value.

## Audit trail

`audit_logs` is append-only and records: actor and role, action, resource type
and id, request id, hashed client IP, outcome and timestamp — for logins, token
refreshes, user creation, every patient/exam/analysis/report access, and every
upload. No clinical content is written to it.

## Transport and network

Terminate TLS at the ingress and never expose the API directly. `CORS_ORIGINS`
is an explicit allow-list (no wildcard). Rate limiting is per identity, with a
Redis-backed shared counter when Celery/Redis is deployed and a per-process
window otherwise; it fails open, because losing the limiter must not take down
the API.

## Secrets

Never in code or in the image. `JWT_SECRET` must be at least 32 random
characters, and the application **refuses to start** in `staging`/`production`
with the development default. Rotating it invalidates all outstanding tokens,
which is the intended behaviour after a compromise.

## LGPD / GDPR posture

The architecture is designed to make compliance possible; compliance itself is
a deployment obligation, not a property of this code.

* *Minimização*: only the fields listed above are stored, and the identity map
  stays in the site's own record system.
* *Finalidade e consentimento*: `consent_research` marks records usable for
  research; enforce it in your cohort queries.
* *Rastreabilidade*: the audit trail and per-inference provenance
  (`model_id`, `model_version`, `input_hash`, `software_version`) support the
  accountability requirement.
* *Direito à eliminação*: deleting a `Patient` cascades to exams, images,
  analyses, runs, results and reports. Object-storage objects must be deleted in
  the same operation — implement that as a documented purge job and verify it,
  since orphaned masks are still patient data.
* *Encarregado/DPO, retention windows, DPIA, breach process*: organizational, and
  outside this repository.

## Not covered here

No clinical validation, no medical-device certification, no formal penetration
test. Before any clinical deployment, add: TLS everywhere, encryption at rest,
key management, backup encryption and restore drills, an intrusion-detection
path, and a documented incident-response process.

## Reporting a vulnerability

Report privately to the maintainers of your deployment. Do not open a public
issue containing patient data, credentials or an exploit against a live system.
