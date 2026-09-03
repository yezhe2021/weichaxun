from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    total = 0
    for filename in ("test_writers.py", "test_losses.py"):
        namespace = runpy.run_path(str(Path(__file__).parent / "tests" / filename))
        tests = sorted((name, value) for name, value in namespace.items() if name.startswith("test_") and callable(value))
        for name, test in tests:
            test()
            total += 1
            print(f"PASS {filename}:{name}", flush=True)
    print(f"{total} unit tests passed", flush=True)


if __name__ == "__main__":
    main()
