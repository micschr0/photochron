# Security policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Use one of these private channels instead:

- GitHub Security Advisories — preferred. Open
  https://github.com/micschr0/photochron/security/advisories/new and we will
  triage from there.
- Email the maintainer privately at the address listed on
  https://github.com/micschr0 (use a subject prefix `[photochron security]`).

We will acknowledge a report within 7 days and aim to ship a fix for confirmed
issues within 30 days. Coordinated disclosure is appreciated; we credit
reporters in the release notes unless asked otherwise.

## Supported versions

photochron is in early alpha (`0.1.x`). Only the latest tagged release on
`main` receives security fixes. Once a stable line ships, this section will
list the supported branches.

## Threat model & user-visible privacy posture

photochron is designed to keep your photo data local. Things to be aware of:

- **All inference is on-device.** No image bytes, EXIF metadata, or
  embeddings are sent to a remote service. The vision LLM runs via a
  *local* Ollama daemon (default `http://localhost:11434`); if you point
  `context.ollama_host` at a remote URL you take on the trust-store of
  that endpoint.
- **GPS extraction is opt-in.** `ingestion.extract_gps` defaults to
  `false` so coordinates do not leak into reports or enriched copies when
  shared. Flip it on only after deciding you're comfortable with that
  trade-off.
- **EXIF-enriched copies embed the full per-photo result JSON** in the
  `UserComment` field (Mode B output, `exif_enriched/`). The JSON
  includes confidence scores and may include matched person hints. If
  you plan to share those copies publicly, consider stripping
  `UserComment` first (`exiftool -UserComment= file.jpg`).
- **`anchors.yaml` contains real birthdays once you fill it in.** Treat
  it like any other private credential file: add it to your local
  `.gitignore`, do not commit it to a public repo, and store backups
  encrypted. The shipped template is fully commented out and contains
  no real data.
- **`.photochron/cache.db` stores face embeddings.** These are biometric
  identifiers and are subject to GDPR / similar regimes in many
  jurisdictions. Treat the cache directory as sensitive personal data
  and consider full-disk encryption.

## Hardening guidance

- Run the pipeline in a dedicated user account when feasible.
- Keep `ollama` and `onnxruntime` up to date — both ship CVE fixes
  occasionally and our Dependabot config covers `pip` and
  `github-actions` dependencies but not external services.
- Do not pipe outputs from untrusted users straight into `photochron
  run --input` unless you trust the file source (image-decoding libs
  have a long history of memory-safety bugs).
