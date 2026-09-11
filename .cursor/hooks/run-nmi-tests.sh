#!/usr/bin/env bash
# Run NMI unit tests after Python/data source edits.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

input="$(cat || true)"
file_path="$(printf '%s' "$input" | "$ROOT/.venv/bin/python" -c '
import json, sys
raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw.strip() else {}
except Exception:
    data = {}
print(data.get("file_path") or data.get("path") or "")
' 2>/dev/null || true)"

case "$file_path" in
  *nmilib.py|*NMI.py|*tests/*|*data.txt|*data1.txt|*data2.txt|*data3.txt|*data4.txt|*data5.txt|*_master.txt|"")
    ;;
  *)
    printf '%s\n' '{}'
    exit 0
    ;;
esac

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

if ! "$PY" -m pytest --version >/dev/null 2>&1; then
  "$PY" - <<'PY'
import json
print(json.dumps({"agent_message": "pytest not available; skipped NMI unit tests."}))
PY
  exit 0
fi

set +e
output="$("$PY" -m pytest -q tests/test_nmi_imputation.py 2>&1)"
status=$?
set -e

FILE_PATH="$file_path" STATUS="$status" OUTPUT="$output" "$PY" - <<'PY'
import json, os
status = int(os.environ.get("STATUS", "1"))
output = os.environ.get("OUTPUT", "")
file_path = os.environ.get("FILE_PATH", "")
lines = output.strip().splitlines()
snippet = "\n".join(lines[-25:])
if status == 0:
    msg = f"NMI unit tests PASSED after edit to {file_path or 'project files'}.\n{snippet}"
else:
    msg = f"NMI unit tests FAILED after edit (exit {status}).\n{snippet}"
print(json.dumps({"agent_message": msg}))
PY

exit 0
