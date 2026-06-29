#!/usr/bin/env bash
set -euo pipefail

ruff format .
ruff check . --fix
