#!/usr/bin/env bash
set -euo pipefail
python -m pip install --disable-pip-version-check -q "pytest>=7.4"
python -m pytest -q
