# PhotoChron

> Local-first CLI tool that sorts digitized family photos without timestamps into chronological order — using on-device AI age estimation, visual context analysis, and user-provided anchor data (birthdays, events).

**All inference runs fully on-device. No data leaves your machine.**

---

## How it works

```mermaid
flowchart LR
    A[📂 Photo folder<br/>no timestamps]:::input --> P[PhotoChron<br/>🔒 local / on-device]:::core
    K[👪 Anchors<br/>birthdays · events]:::input --> P
    P --> O[📅 Chronologically<br/>sorted output]:::output

    classDef input fill:#f4e6c8,stroke:#8a6f3a,stroke-width:2px,color:#3a2e14
    classDef core fill:#c9a961,stroke:#6b4f1f,stroke-width:3px,color:#1f1608
    classDef output fill:#e8d5a8,stroke:#8a6f3a,stroke-width:2px,color:#3a2e14
```

Under the hood, PhotoChron runs a 6-stage pipeline. Each stage writes to a local SQLite feature store, so expensive AI inference runs only once per photo:

```mermaid
flowchart LR
    I[1 · Ingestion<br/>📥 scan · hash · EXIF]:::stage
    F[2 · Face Layer<br/>👤 detect · age · match]:::stage
    C[3 · Context Layer<br/>🖼️ decade · season · medium]:::stage
    A[4 · Anchor Layer<br/>🎂 birthdays · events]:::stage
    R[5 · Ranking Engine<br/>⚖️ fuse · constrain]:::stage
    O[6 · Output Layer<br/>📤 sorted copies]:::stage
    DB[(🗄️ SQLite<br/>Feature Store)]:::store

    I --> F --> C --> A --> R --> O
    I -.-> DB
    F -.-> DB
    C -.-> DB
    A -.-> DB
    R -.-> DB
    O -.-> DB

    classDef stage fill:#f4e6c8,stroke:#6b4f1f,stroke-width:2px,color:#1f1608
    classDef store fill:#5c4a2a,stroke:#2a1e0a,stroke-width:2px,color:#f4e6c8
```

See [docs/pipeline.md](docs/pipeline.md) for a deep dive into each stage.

---

## What you get

PhotoChron never touches your originals. It writes copies into `{output_dir}/` with two complementary layouts:

```
photochron_output/
├── renamed/                                  ← Mode A: chronologically sorted filenames
│   ├── 0001_1987-est_scan_0042.jpg           (prefix = sort rank, then estimated year)
│   ├── 0002_1987-est_scan_0017.jpg
│   ├── 0003_1988-est_scan_0091.jpg
│   └── ...
├── exif_enriched/                            ← Mode B: original names, enriched EXIF
│   ├── scan_0042.jpg                         (DateTimeOriginal = 1987:01:01, UserComment = result JSON)
│   ├── scan_0017.jpg
│   └── ...
├── photochron_report.json                    ← per-photo confidence, anchors used, flags
└── photochron_timeline.csv                   ← flat timeline for spreadsheets
```

**Mode A** lets you drop the `renamed/` folder into any photo viewer and browse your family history in order.
**Mode B** preserves original filenames but writes the estimated date into EXIF, so Apple Photos / Lightroom / digiKam pick it up automatically.

Photos flagged with low confidence end up in `review_needed = true` in `photochron_report.json` — PhotoChron surfaces uncertainty rather than guessing silently.

---

## ⚠️ Alpha status

PhotoChron is in **early alpha**. The following CLI commands are currently **demo stubs** and do not yet execute the full pipeline:

- `photochron run` — simulates pipeline progress (no real processing)
- `photochron cluster` — placeholder
- `photochron rerun` — placeholder

The `photochron status` command is functional and reads the SQLite feature store. The underlying pipeline modules (`src/photochron/pipeline/`, `src/photochron/context/`, `src/photochron/face/`, etc.) are implemented and unit-tested, but not yet wired into the CLI. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Installation

Requires **Python 3.12+** and [Ollama](https://ollama.com) (for the local vision LLM).

```bash
git clone https://github.com/micschr0/image-age-sorter.git
cd image-age-sorter

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

For local model setup (Ollama, InsightFace), see [docs/ollama-setup.md](docs/ollama-setup.md).

---

## Quick start

```bash
# Show pipeline status (functional)
python -m photochron status

# Full pipeline run (currently a demo stub — see Alpha status above)
python -m photochron run --input ./photos --output ./photochron_output

# Dry run (no file writes)
python -m photochron run --input ./photos --dry-run
```

Configuration lives in `config.yaml`; anchor data (persons, birthdays, events) in `anchors.yaml`.
See [docs/configuration.md](docs/configuration.md) for all options.

---

## Documentation

- [Pipeline architecture](docs/pipeline.md) — detailed 6-stage walkthrough
- [Configuration reference](docs/configuration.md) — all `config.yaml` options
- [Ollama setup](docs/ollama-setup.md) — installing the local vision LLM
- [Testing](docs/testing.md) — test suite layout and conventions
- [Changelog](docs/CHANGELOG.md)

---

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test workflow, and coding standards.

---

## License

MIT — see [LICENSE](LICENSE).
