# Configuration Reference

photochron uses a hierarchical configuration system with sensible defaults. Configuration can be customized via `config.yaml` or environment variables.

## Configuration Structure

### Root Configuration (`Config` class)
- `version`: Configuration schema version (string)
- `paths`: Path configuration (`ConfigPaths`)
- `models`: AI model configuration (`ConfigModels`)
- `ingestion`: Ingestion stage configuration (`ConfigIngestion`)
- `face`: Face layer configuration (`ConfigFace`)
- `pipeline`: Pipeline configuration (`ConfigPipeline`)
- `context`: Context layer configuration (`ConfigContext`)

### Path Configuration (`ConfigPaths`)
- `cache_dir`: Directory for cache and database (default: `.photochron`)
- `thumbs_dir`: Directory for downsampled thumbnails (default: `.photochron/thumbs`)
- `output_dir`: Directory for output files (default: `photochron_output`)

### Model Configuration (`ConfigModels`)
- `insightface_version`: InsightFace model version (default: `buffalo_l`)
- `ollama_model`: Ollama vision LLM model (default: `llava-next:7b`)
- `fallback_model`: Fallback vision model (default: `moondream2`)
- `max_image_size`: Maximum image size for processing in pixels (default: `1024`)

### Ingestion Configuration (`ConfigIngestion`)
- `max_downsample_size`: Maximum size for downsampled images in pixels (default: `1024`)
- `supported_formats`: List of supported image file extensions (default: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.cr2`, `.nef`, `.arw`, `.dng`)
- `skip_duplicates`: Whether to skip duplicate files (default: `true`)
- `extract_gps`: Whether to extract GPS coordinates from EXIF (default: `false`, opt-in – GPS can de-anonymize private family photos when reports are shared)
- `workers`: Number of concurrent threads used to decode images, compute hashes, and extract EXIF (default: `4`, range 1–32). Ingestion releases the GIL inside Pillow / imagehash / sqlite3, so threads give near-linear speed-up on 4–8 cores. Set to `1` for deterministic ordering when debugging.
- `fallback_timestamp`: Fallback timestamp source when EXIF missing (default: `file_mtime`)

### Face Layer Configuration (`ConfigFace`)
- `model_name`: InsightFace model name (default: `""` – opt-in; uncomment in `config.yaml` after reviewing the model's license)
- `detection_threshold`: Minimum confidence for face detection (0.0-1.0, default: `0.5`)
- `matching_threshold`: Cosine similarity threshold for person matching (0.0-1.0, default: `0.6`)
- `age_confidence_scale`: Scale factor for age estimation standard deviation (default: `0.1`)
- `backend`: ONNX Runtime execution backend (default: `"auto"`). One of:
  - `"auto"` – CoreML on arm64 macOS (Apple Silicon), CPU elsewhere
  - `"cpu"` – always CPU (works everywhere)
  - `"cuda"` – NVIDIA GPU via the CUDA EP (requires a CUDA-enabled `onnxruntime` build)
  - `"coreml"` – Apple Neural Engine / GPU / CPU via the CoreML EP. Provider options use `MLProgram` + `MLComputeUnits=ALL` so the runtime can partition per op. Falls back to CPU with a warning when the CoreML EP is missing from the installed `onnxruntime` build (this is the case for the **official `onnxruntime` wheel on macOS arm64**; install a wheel with the CoreML EP such as [`onnxruntime-silicon`](https://github.com/cansik/onnxruntime-silicon) or build from source with `--use_coreml`).

  Run `photochron doctor` to see which providers ONNX Runtime actually exposes on your host and how `"auto"` is resolved – the command warns when you are on Apple Silicon but the CoreML EP is not available.
- `use_gpu`: **Deprecated.** Prefer `backend`. Setting `use_gpu: true` while leaving `backend` at `"auto"` is migrated to `backend: "cuda"` for backward compatibility.
- `batch_size`: Batch size for face detection (default: `1`)

### Pipeline Configuration (`ConfigPipeline`)
- `face_age_weight`: Weight for face age estimates in ranking (0.0-1.0, default: `0.45`)
- `llm_decade_weight`: Weight for LLM decade estimates in ranking (0.0-1.0, default: `0.30`)
- `photo_medium_weight`: Weight for photo medium priors in ranking (0.0-1.0, default: `0.10`)
- `min_confidence_threshold`: Minimum confidence for results to be considered reliable (0.0-1.0, default: `0.5`)
- `max_pairwise_comparisons`: Maximum number of pairwise LLM comparisons per run (default: `500`)

### Context Layer Configuration (`ConfigContext`)
The context layer configuration provides comprehensive settings for Ollama integration, graceful degradation, and resource management:

#### Ollama Server Settings
- `ollama_host`: Ollama server URL (default: `http://localhost:11434`)
- `ollama_timeout`: Timeout in seconds for Ollama requests (default: `300`)

#### Model Management
- `primary_model`: Primary vision LLM model (default: `llava-next:7b`)
- `fallback_model`: Fallback vision model (default: `moondream2`)

#### Retry and Error Handling
- `max_retries`: Maximum retry attempts for LLM failures (default: `3`)
- `retry_delay`: Delay between retries in seconds (default: `2.0`)
- `use_fallback_on_failure`: Use fallback strategies on analysis failure (default: `true`)
- `store_minimal_on_complete_failure`: Store minimal data when analysis completely fails (default: `true`)

#### Ollama Runtime Tuning (Apple Silicon / performance)
These fields are forwarded directly to `ollama.generate(...)`. Tuning them is the single biggest Apple-Silicon throughput win.

- `keep_alive`: Duration string (`"30m"`, `"1h"`, `"-1"` for forever) controlling how long Ollama holds the model in memory between photos. Default: `"30m"`. Without a long `keep_alive`, Ollama reloads the ~5 GB llava-next weights every few minutes of idle time, wiping any speedup from Metal.
- `num_ctx`: Context-window size in tokens (default: `2048`). Lower values reduce Metal memory pressure on 8–16 GB machines; raise only if your prompts or outputs get truncated.
- `num_gpu`: Number of model layers to offload to the GPU. `-1` (default) means auto — on Apple Silicon this puts all layers on Metal. Set to `0` to force CPU.
- `model_options`: Per-model overrides as a nested mapping. Useful when a lighter fallback model (e.g. `moondream2`) can use a smaller context:
  ```yaml
  context:
    num_ctx: 2048
    keep_alive: "30m"
    model_options:
      moondream2:
        num_ctx: 1024
      "llava-next:7b":
        keep_alive: "1h"
  ```
  Any key supported by Ollama's `options` dict works; `keep_alive` is handled as a special top-level override.

During long generations the CLI now logs a heartbeat line every ~5s so the terminal does not appear frozen while the model is working.

#### Memory Management
- `memory_warning_threshold_mb`: Memory warning threshold in MB. Logs warning if available memory falls below this value. Must be greater than `memory_critical_threshold_mb`. (default: `100`, range: 10-10000)
- `memory_critical_threshold_mb`: Memory critical threshold in MB. Skips batch processing if available memory falls below this value. Must be less than `memory_warning_threshold_mb`. (default: `50`, range: 10-10000)
- `memory_retry_delay_seconds`: Delay in seconds to wait when memory is critically low before retrying batch processing. (default: `30`, range: 1-300)

#### Processing Settings
- `batch_size`: Batch size for processing images (default: `1`)
- `min_decade_confidence`: Minimum confidence for decade estimates (0.0-1.0, default: `0.3`)
- `min_season_confidence`: Minimum confidence for season estimates (0.0-1.0, default: `0.4`)

## Configuration Validation

The context layer performs comprehensive configuration validation:

1. **Model Availability Check**: Verifies that configured models are available in Ollama
2. **Server Health Check**: Validates Ollama server connectivity
3. **Graceful Degradation**: Automatically falls back to available models or enters degraded mode
4. **Runtime Health Monitoring**: Provides real-time health status via `health_status` property

## Environment Variables

Any configuration value can be overridden using environment variables. The format is:
```
PHOTOCHRON_<SECTION>_<KEY>
```

### Examples:
```bash
# Override Ollama host
export PHOTOCHRON_CONTEXT_OLLAMA_HOST="http://192.168.1.100:11434"

# Increase timeout
export PHOTOCHRON_CONTEXT_OLLAMA_TIMEOUT=600

# Use different model
export PHOTOCHRON_CONTEXT_PRIMARY_MODEL="llava-next:13b"

# Disable GPS extraction
export PHOTOCHRON_INGESTION_EXTRACT_GPS=false
```

## Default Configuration File

```yaml
# photochron Configuration
# Default values from architecture specification

version: "1.0"

paths:
  cache_dir: ".photochron"
  thumbs_dir: ".photochron/thumbs"
  output_dir: "photochron_output"

models:
  insightface_version: "buffalo_l"
  ollama_model: "llava-next:7b"
  fallback_model: "moondream2"
  max_image_size: 1024

ingestion:
  max_downsample_size: 1024
  supported_formats:
    - ".jpg"
    - ".jpeg"
    - ".png"
    - ".heic"
    - ".heif"
    - ".cr2"
    - ".nef"
    - ".arw"
    - ".dng"
  skip_duplicates: true
  extract_gps: false  # opt-in; see license/privacy notes
  fallback_timestamp: "file_mtime"

face:
  # model_name: "buffalo_l"  # opt-in; uncomment after license review
  detection_threshold: 0.5
  matching_threshold: 0.6
  age_confidence_scale: 0.1
  use_gpu: false
  batch_size: 1

context:
  ollama_host: "http://localhost:11434"
  ollama_timeout: 300
  max_retries: 3
  retry_delay: 2.0
  primary_model: "llava-next:7b"
  fallback_model: "moondream2"
  batch_size: 1
  min_decade_confidence: 0.3
  min_season_confidence: 0.4
  use_fallback_on_failure: true
  store_minimal_on_complete_failure: true
  memory_warning_threshold_mb: 100
  memory_critical_threshold_mb: 50
  memory_retry_delay_seconds: 30

pipeline:
  face_age_weight: 0.45
  llm_decade_weight: 0.30
  photo_medium_weight: 0.10
  min_confidence_threshold: 0.5
  max_pairwise_comparisons: 500
```

## Health Status Monitoring

The context layer provides real-time health status through the `health_status` property:

```python
{
    "is_healthy": True,  # Overall health status
    "degraded_mode": False,  # Whether operating in degraded mode
    "available_models": {  # Dictionary of available models
        "primary": True,
        "fallback": True
    }
}
```

**Degraded Mode**: When neither primary nor fallback models are available, the system enters degraded mode. In this mode:
- Context analysis is skipped
- Minimal data may be stored (if `store_minimal_on_complete_failure` is `true`)
- The pipeline continues with other stages