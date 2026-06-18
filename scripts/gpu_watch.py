from __future__ import annotations

import argparse
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print nvidia-smi utilization periodically.")
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end = time.time() + args.seconds
    query = "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    while time.time() < end:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
        )
        print(result.stdout.strip() or result.stderr.strip(), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
