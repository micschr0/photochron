#!/usr/bin/env python3
"""
Profile face detection performance for photochron face layer.

Measures detection time on CPU (and optionally GPU) with realistic image sizes.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photochron.face.insightface_wrapper import InsightFaceWrapper


def create_test_image(width=1024, height=768):
    """Create a synthetic test image with random colors."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def load_real_image(image_path):
    """Load an image from disk."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return np.array(img)


def profile_detection(wrapper, image, num_runs=10, warmup=2):
    """Profile face detection on a given image."""
    times = []
    for i in range(warmup + num_runs):
        start = time.perf_counter()
        detections = wrapper.detect_faces(image)
        end = time.perf_counter()
        if i >= warmup:
            times.append(end - start)
    return times, detections


def main():
    parser = argparse.ArgumentParser(description="Profile face detection performance")
    parser.add_argument("--image", type=Path, help="Path to test image (optional)")
    parser.add_argument("--width", type=int, default=1024, help="Image width for synthetic image")
    parser.add_argument("--height", type=int, default=768, help="Image height for synthetic image")
    parser.add_argument("--runs", type=int, default=10, help="Number of measurement runs")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    # Load or create test image
    if args.image and args.image.exists():
        print(f"Loading image: {args.image}")
        test_image = load_real_image(args.image)
    else:
        print(f"Creating synthetic image {args.width}x{args.height}")
        test_image = create_test_image(args.width, args.height)

    print(f"Image shape: {test_image.shape}, dtype: {test_image.dtype}")

    # Create wrapper
    wrapper = InsightFaceWrapper(
        model_name="buffalo_l",
        detection_threshold=args.threshold,
        use_gpu=args.gpu,
    )
    print(f"Wrapper initialized (GPU: {args.gpu})")

    # Warm-up model loading
    print("Loading model (first call)...")
    wrapper.detect_faces(test_image)
    print("Model loaded.")

    # Profile detection
    print(f"Profiling detection ({args.runs} runs after {args.warmup} warmup)...")
    times, detections = profile_detection(wrapper, test_image, args.runs, args.warmup)

    # Print results
    print(f"\nDetection results: {len(detections)} faces found")
    for i, (bbox, conf) in enumerate(detections):
        print(f"  Face {i + 1}: bbox {bbox}, confidence {conf:.3f}")

    print("\nTiming results (seconds):")
    print(f"  Min: {min(times):.4f}")
    print(f"  Max: {max(times):.4f}")
    print(f"  Mean: {statistics.mean(times):.4f}")
    print(f"  StdDev: {statistics.stdev(times):.4f}")
    print(f"  Median: {statistics.median(times):.4f}")
    print(f"  Total: {sum(times):.4f}")

    # Compute throughput (images per second)
    mean_time = statistics.mean(times)
    if mean_time > 0:
        print(f"\nThroughput: {1 / mean_time:.2f} images/sec")

    # Memory usage (rough estimate)
    try:
        import psutil

        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"Process memory: {mem_mb:.1f} MB")
    except ImportError:
        print("psutil not installed, skipping memory reporting")

    # GPU info (if applicable)
    if args.gpu:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            print(f"Available ONNX providers: {providers}")
        except ImportError:
            print("ONNX Runtime not available")


if __name__ == "__main__":
    main()
