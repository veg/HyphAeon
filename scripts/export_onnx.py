#!/usr/bin/env python
"""
scripts/export_onnx.py
----------------------
WHY THIS FILE EXISTS

Thin wrapper so the ONNX export can be run as a script from a checkout
(`python scripts/export_onnx.py --variant all`) without the console entry
point; identical to `hyphaeon export-onnx`. All logic lives in
hyphaeon/export.py so the CLI subcommand and this script cannot drift.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hyphaeon.export import main  # noqa: E402

if __name__ == "__main__":
    main()
