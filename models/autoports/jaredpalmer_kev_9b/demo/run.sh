#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${1:-http://127.0.0.1:8008}"
PORT="${DEMO_PORT:-8080}"
BIND="${DEMO_BIND:-127.0.0.1}"
if ! python3 -c "import socket, sys; s = socket.socket(); s.bind((sys.argv[1], int(sys.argv[2])))" "${BIND}" "${PORT}" 2>/dev/null; then
  echo "port ${PORT} on ${BIND} is busy; pick another with DEMO_PORT=<port> $0 ${SERVER}" >&2
  exit 1
fi
echo "kev-9b demo: serving ${DEMO_DIR} on http://${BIND}:${PORT}/"
echo "Open: http://127.0.0.1:${PORT}/#server=${SERVER}"
echo "Stop with Ctrl+C."
exec python3 -m http.server "${PORT}" --bind "${BIND}" --directory "${DEMO_DIR}"
