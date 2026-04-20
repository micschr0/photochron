Models & Hardware

Apple Silicon Execution Model
The M3 MacBook Air uses Unified Memory – CPU, GPU, and Neural Engine share the same 16 GB.
This means there is no VRAM copy overhead; models loaded into RAM are directly accessible
by the ANE and GPU.

Face Layer: InsightFace buffalo_l
Property             | Value
Task                 | Face detection + embedding + age estimate
Format               | ONNX
Execution            | ONNX Runtime + CoreML ExecutionProvider
Hardware target      | ANE / GPU via CoreML
Latency              | 100–300ms/image on M3
Memory footprint     | ~200–400 MB
Age MAE              | ±3–5 years (adults), ±4–7 years (children)

Why InsightFace over DeepFace:
Superior accuracy on low-resolution historical photos (arXiv 2511.14689).
Single forward pass for detection + embedding + age (no pipeline glue needed).

Fallback: CPU execution if CoreML provider unavailable.

Context Layer: Ollama (MLX backend)
Property         | Primary              | Fallback
Model            | llava-next:7b        | moondream2 (1.8B)
Backend          | Ollama + MLX         | Ollama + MLX
Hardware target  | GPU via Metal/MLX    | GPU via Metal/MLX
Latency          | ~2–5s/image          | ~0.5–1.5s/image
Memory footprint | ~4.5 GB              | ~1.5 GB
Output quality   | High (decade, event) | Sufficient (decade)

Fallback trigger: available RAM < 8 GB at runtime → switch to moondream2.

MLX note: Ollama adopted MLX as its Apple Silicon backend (March 2026).
Keep Ollama updated to benefit from ongoing MLX performance improvements.
Do NOT run Ollama inside Docker on Apple Silicon – GPU passthrough is unsupported;
always run as native macOS process.

Memory Budget (16 GB system)
Component              | Footprint
InsightFace buffalo_l  | ~400 MB
llava-next:7b (Ollama) | ~4.5 GB
SQLite + Python        | ~300 MB
macOS overhead         | ~3 GB
**Total**              | **~8.2 GB**
**Free headroom**      | **~7.8 GB**

Both models fit comfortably in 16 GB without swapping.

Changing Models

- Switching default LLM model invalidates all `context` cache rows → rerun Stage 3.
- Document model version in `pipeline_runs` table on every run.
- ⚠️ Ask before switching default model (see `boundaries.md`).