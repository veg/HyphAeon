"""
hyphaeon/io.py
--------------
Shared output helpers: directory creation, JSON/CSV writing with confirmation
prints, and p/q value formatting for pretty-printing.
"""

import os
import json

import pandas as pd


def ensure_parent_directory(path):
    """Create the parent directory of `path` if it doesn't exist."""
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)


def write_json(path, data, label="JSON results"):
    """Write `data` as JSON to `path` and print a confirmation message.

    `label` is used in the confirmation print (e.g. "JSON results", "CSV summary").
    """
    ensure_parent_directory(path)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[✓] {label} written to: {path}")


def write_csv(path, df_or_records, label="CSV results"):
    """Write a DataFrame (or list of dicts) to CSV and print confirmation.

    If `df_or_records` is a list of dicts, it's converted to a DataFrame first.
    """
    ensure_parent_directory(path)
    if isinstance(df_or_records, list):
        df_or_records = pd.DataFrame(df_or_records)
    df_or_records.to_csv(path, index=False)
    print(f"[✓] {label} written to: {path}")


def format_pq(value):
    """Format a p-value or q-value for pretty-printing.

    Uses scientific notation for small values (< 0.01), fixed notation otherwise.
    """
    return f"{value:.2e}" if value < 0.01 else f"{value:.3f}"
