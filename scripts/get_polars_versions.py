#!/usr/bin/env python3
"""Get polars versions >= 1.0.0 from pip index."""

import sys
import json

data = json.load(sys.stdin)
versions = [v for v in data["versions"] if tuple(map(int, v.split("."))) >= (1, 2, 1)]
versions.sort(key=lambda x: tuple(map(int, x.split("."))), reverse=True)
print("\n".join(versions))
