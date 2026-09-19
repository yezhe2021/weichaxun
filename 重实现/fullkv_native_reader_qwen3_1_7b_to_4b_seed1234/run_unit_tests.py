from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    namespace = runpy.run_path(str(Path(__file__).parent / "tests" / "test_writers.py"))
    tests = sorted((name, value) for name, value in namespace.items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}", flush=True)
    print(f"{len(tests)} Writer unit tests passed", flush=True)


if __name__ == "__main__":
    main()
