#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
.venv/bin/ruff check src tests
.venv/bin/pytest
