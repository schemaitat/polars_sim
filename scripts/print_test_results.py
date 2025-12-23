#!/usr/bin/env python3
"""Print test results table from matrix tests."""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: print_test_results.py <tmpdir> <version1> [version2] ...")
        sys.exit(1)

    tmpdir = Path(sys.argv[1])
    versions = sys.argv[2:]

    # Print results table
    print()
    print("╔════════════╦═════════╦══════════════╗")
    print("║  Version   ║ Status  ║ Time (sec)   ║")
    print("╠════════════╬═════════╬══════════════╣")

    failed_versions = []
    for version in versions:
        status_file = tmpdir / f"{version}.status"
        if status_file.exists():
            with open(status_file) as f:
                parts = f.read().strip().split()
                status = parts[0]
                duration = parts[1] if len(parts) > 1 else "N/A"

            if status == "SUCCESS":
                print(f"║ {version:10s} ║ ✓ Pass  ║ {duration:>12s} ║")
            else:
                print(f"║ {version:10s} ║ ✗ Fail  ║ {duration:>12s} ║")
                failed_versions.append(version)
        else:
            print(f"║ {version:10s} ║ ✗ Error ║          N/A ║")
            failed_versions.append(version)

    print("╚════════════╩═════════╩══════════════╝")

    # Show first failed test log if any
    if failed_versions:
        print()
        print("Sample error from first failed test:")
        version = failed_versions[0]
        log_file = tmpdir / f"{version}.log"
        if log_file.exists():
            print(f"=== Log for version {version} ===")
            print(log_file.read_text())


if __name__ == "__main__":
    main()
