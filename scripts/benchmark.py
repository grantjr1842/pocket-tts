import json
import statistics
import subprocess
import sys
from typing import Dict, List


def run_benchmark(iterations: int = 5):
    """Run pocket-tts generate multiple times and collect metrics."""
    print(f"Running benchmark with {iterations} iterations...")

    metrics: Dict[str, List[float]] = {}

    full_cmd = [
        sys.executable,
        "-m",
        "pocket_tts",
        "generate",
        "--text",
        "The quick brown fox jumps over the lazy dog.",
        "--output-path",
        "/dev/null",
    ]

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}...", end="", flush=True)

        result = subprocess.run(
            full_cmd, capture_output=True, text=True, env={"PYTHONPATH": "."}
        )

        if result.returncode != 0:
            print("FAILED")
            print(result.stderr)
            continue

        print("Done")

        # Parse output for metrics
        # We expect JSON outputs in stderr/stdout due to structlog
        for line in result.stderr.splitlines() + result.stdout.splitlines():
            try:
                log_entry = json.loads(line)
                if log_entry.get("event") == "performance_metric":
                    op = log_entry.get("operation")
                    duration = log_entry.get("duration_seconds")
                    if op and duration:
                        if op not in metrics:
                            metrics[op] = []
                        metrics[op].append(duration)
            except json.JSONDecodeError:
                continue

    return metrics


def print_dashboard(metrics: Dict[str, List[float]]):
    """Print a simple ASCII dashboard of the metrics."""
    print("\n" + "=" * 50)
    print("PERFORMANCE DASHBOARD")
    print("=" * 50)

    if not metrics:
        print("No metrics collected.")
        return

    print(
        f"{'Operation':<25} | {'Mean (s)':<10} | {'Median (s)':<10} | {'Stdev (s)':<10}"
    )
    print("-" * 65)

    for op, values in metrics.items():
        if not values:
            continue

        mean = statistics.mean(values)
        median = statistics.median(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0

        print(f"{op:<25} | {mean:<10.4f} | {median:<10.4f} | {stdev:<10.4f}")

    print("-" * 65)


if __name__ == "__main__":
    metrics = run_benchmark()
    print_dashboard(metrics)
