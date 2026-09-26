#!/usr/bin/env bash
# =============================================================================
# One-command setup for the arXiv Digest & QA Agent.
#
#   ./setup.sh
#
# Creates a local virtual environment in .venv, installs all dependencies, and
# creates a .env from the template if you don't have one yet. Safe to re-run.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Using interpreter: $($PYTHON --version 2>&1)"

if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment (.venv)"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> Installing dependencies"
python -m pip install --quiet -r requirements.txt

if [ ! -f ".env" ]; then
  echo "==> Creating .env from .env.example (fill in an LLM key or use ollama)"
  cp .env.example .env
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .env and set ONE provider (GROQ_API_KEY, GEMINI_API_KEY, or ollama)."
echo "  2. Activate the environment:   source .venv/bin/activate"
echo "  3. Run the agent:              python main.py digest \"attention is all you need\""
echo ""
