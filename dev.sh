#!/usr/bin/env bash
#
# One-time setup to make this script executable:
#   chmod +x "/Users/akhileshbharatham/Documents/GitHub/PulmoSim.py/dev.sh"
#
# Run both backend (FastAPI) and frontend (Vite) together:
#   "/Users/akhileshbharatham/Documents/GitHub/PulmoSim.py/dev.sh"
#
# Optional custom ports (defaults: backend 8000, frontend 5173):
#   PORT=9000 VITE_PORT=5174 /Users/akhileshbharatham/Documents/GitHub/PulmoSim.py/dev.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
VITE_PORT="${VITE_PORT:-5173}"

# Optional: activate local venv if present
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
fi

# Free backend port if taken
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | xargs -r kill || true

echo "Starting backend on :$PORT ..."
PYTHONPATH="$ROOT_DIR" \
uvicorn server:app --host 0.0.0.0 --port "$PORT" --reload --app-dir "$ROOT_DIR" &
BACK_PID=$!

cleanup() {
  echo "\nShutting down..."
  kill $BACK_PID $FRONT_PID 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Starting frontend on :$VITE_PORT (API http://localhost:$PORT) ..."
cd "$ROOT_DIR/frontend"
VITE_API_BASE="http://localhost:$PORT" npm run dev &
FRONT_PID=$!

wait


