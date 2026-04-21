# PhotoChron Documentation

This directory contains user-facing documentation for PhotoChron.

## Documentation Index

### User Guides
- **[Pipeline Architecture](pipeline.md)**: Detailed explanation of the 6-stage pipeline
- **[Configuration Reference](configuration.md)**: Complete configuration options and environment variables
- **[Testing Strategy](testing.md)**: Comprehensive test suite documentation and test organization
- **[Changelog](CHANGELOG.md)**: Version history and release notes

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd image-age-sorter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Basic Usage
```bash
# Full pipeline run
python -m photochron run --input ./photos --output ./photochron_output

# Dry run (no file writes)
python -m photochron run --input ./photos --dry-run

# Show cache stats
python -m photochron status
```

## Configuration

PhotoChron uses a hierarchical configuration system. The default configuration is in `config.yaml` at the project root. Any value can be overridden using environment variables:

```bash
# Override Ollama host
export PHOTOCHRON_CONTEXT_OLLAMA_HOST="http://192.168.1.100:11434"

# Increase timeout
export PHOTOCHRON_CONTEXT_OLLAMA_TIMEOUT=600
```

See [Configuration Reference](configuration.md) for complete details.

## Architecture Overview

PhotoChron uses a 6-stage pipeline where each stage reads/writes to a SQLite Feature Store only. Stages are independently re-runnable.

**Pipeline Stages:**
1. **Ingestion** → Reads image files, extracts metadata, creates downsampled copies
2. **Face Layer** → Detects faces, computes embeddings, estimates ages, matches persons
3. **Context Layer** → Analyzes visual context using vision LLM (Ollama) with graceful degradation
4. **Anchor Layer** → Applies user-provided constraints (birthdays, events)
5. **Ranking Engine** → Combines evidence, applies constraints, produces final order
6. **Output Layer** → Creates renamed copies and EXIF-enriched versions

See [Pipeline Architecture](pipeline.md) for detailed stage-by-stage documentation.