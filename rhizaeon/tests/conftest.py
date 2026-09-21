"""Shared fixtures and environment setup for rhizaeon tests."""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
