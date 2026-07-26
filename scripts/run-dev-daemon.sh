#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec .venv/bin/virtual-human >> runtime/gateway-daemon.log 2>&1
