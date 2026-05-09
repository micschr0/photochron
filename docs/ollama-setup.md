# Ollama Setup

photochron's Context Layer uses a local [Ollama](https://ollama.com) server for vision LLM inference. Nothing is sent to external services — model weights run on your machine. Ollama 0.19 and newer ship an MLX backend preview on Apple Silicon (Ollama blog post from March 2026) that roughly doubles decode speed compared to the legacy llama.cpp/Metal path that older Ollama versions used; on Linux/Windows the llama.cpp backend still applies. This document covers installation, model pulls, verification, and common issues.

## Install Ollama

### macOS (Apple Silicon recommended; MLX-accelerated in Ollama ≥ 0.19, llama.cpp/Metal otherwise)

```bash
brew install ollama
# or: download the signed .dmg from https://ollama.com/download
```

Start the background service:

```bash
brew services start ollama
# or run in the foreground:
ollama serve
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The installer registers a `systemd` unit. Verify it's running:

```bash
systemctl status ollama
```

### Windows

Download the installer from [ollama.com/download](https://ollama.com/download) and launch Ollama. The tray app starts the local server on `http://localhost:11434`.

## Pull the Required Models

photochron uses a primary and a fallback vision model. Defaults:

- **Primary**: `llava-next:7b` — higher quality, ~5 GB
- **Fallback**: `moondream2` — smaller, faster, ~1.7 GB

Pull both:

```bash
ollama pull llava-next:7b
ollama pull moondream2
```

Tip: you can pull just one if disk or RAM is tight. photochron will detect which models are available at startup and degrade gracefully.

List installed models:

```bash
ollama list
```

## Verify the Setup

1. **Server reachable**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   Should return JSON with your installed models.

2. **Smoke test from Ollama**:
   ```bash
   ollama run moondream2 "Say hello"
   ```

3. **Smoke test from photochron**:
   ```bash
   python -m photochron status
   ```
   Look for `context_layer: healthy` (or a warning if a model is missing).

4. **Run a dry context pass**:
   ```bash
   python -m photochron run --dry-run --stages context_layer
   ```
   Inspect the logs for health and progress lines.

## Configure photochron

Set the server and model names in `config.yaml`:

```yaml
context:
  ollama_host: http://localhost:11434
  primary_model: llava-next:7b
  fallback_model: moondream2
  ollama_timeout: 300
```

Full option list: see `docs/context-layer.md` and `examples/context-config-example.yaml`.

## Remote / Non-Default Server

If Ollama runs on another machine or a non-default port, update the host:

```yaml
context:
  ollama_host: http://192.168.1.42:11434
```

> Security note: photochron is local-first by design. Only point it at a trusted private server — biometric family data must not leave your network.

## Lighter Configuration (low-RAM / older hardware)

```yaml
context:
  primary_model: moondream2
  fallback_model: moondream2
  batch_size: 1
  ollama_timeout: 600
  memory_warning_threshold_mb: 200
  memory_critical_threshold_mb: 100
```

This skips the larger 7B model entirely and gives Ollama extra time on slower hardware.

## Alternative Models

Any Ollama-compatible vision model can be used as long as it implements the chat API with image attachments. Known working choices:

- `llava:13b` — higher quality, higher memory
- `llava-next:7b` — default primary
- `moondream2` — default fallback, smallest
- `bakllava` — older but lightweight

Set the model names in `config.yaml` and make sure they match the tags you pulled.

## Troubleshooting

### `connection refused` / `Ollama server unavailable`
- Is the Ollama service running? `brew services list` (macOS) or `systemctl status ollama` (Linux).
- Is the port blocked? `lsof -i :11434` should show `ollama`.
- Is `ollama_host` set correctly in `config.yaml`?

### `model 'llava-next:7b' not found`
- Pull it: `ollama pull llava-next:7b`.
- Check exact tag: `ollama list` — model names in `config.yaml` must match.

### `Request timed out`
- Increase `ollama_timeout` (default 300s) in `config.yaml`.
- Consider switching `primary_model` to `moondream2` on slower hardware.

### `OOM` / system becomes unresponsive
- Lower `batch_size` to `1`.
- Switch to `moondream2`.
- Raise `memory_critical_threshold_mb` — photochron will pause batches when RAM is tight.

### Models load every run and slow things down
- Keep Ollama resident (`brew services start ollama` / systemd unit) so models stay warm.
- Avoid toggling `primary_model` between runs.

## Uninstall / Reset

```bash
ollama rm llava-next:7b
ollama rm moondream2

# macOS
brew services stop ollama
brew uninstall ollama

# Linux
sudo systemctl stop ollama
sudo systemctl disable ollama
# remove the binary per the installer's instructions
```

Models are stored in `~/.ollama/models` — remove that directory to reclaim disk space if needed.
