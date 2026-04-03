#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$DEFAULT_ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
	set -a
	source "$ENV_FILE"
	set +a
fi

ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT_DIR}"
LOG_DIR="$ROOT_DIR/backend/logs"
LOG_FILE="$LOG_DIR/backend.nohup.log"
PID_FILE="$LOG_DIR/backend.pid"

# Priority: explicit python path > conda env name > system python.
BACKEND_PYTHON_BIN="${BACKEND_PYTHON_BIN:-}"
BACKEND_CONDA_ENV="${BACKEND_CONDA_ENV:-pytorch_yc}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

START_TS="$(date '+%F %T')"

if [[ -f "$ENV_FILE" ]]; then
    ENV_STATUS="loaded"
else
    ENV_STATUS="missing"
fi

if [[ -f "$PID_FILE" ]]; then
	OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
	if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
		{
			echo ""
			echo "[$START_TS] backend launch skipped: already running"
			echo "[$START_TS] existing pid: $OLD_PID"
			echo "[$START_TS] log file: $LOG_FILE"
		} >> "$LOG_FILE"
		echo "Backend is already running with PID=$OLD_PID"
		echo "Log: $LOG_FILE"
		exit 0
	fi
fi

LAUNCH_MODE="system-python"
CMD_DESC="python backend/main.py"
if [[ -n "$BACKEND_PYTHON_BIN" ]]; then
	LAUNCH_MODE="python-bin"
	CMD_DESC="$BACKEND_PYTHON_BIN backend/main.py"
	nohup "$BACKEND_PYTHON_BIN" backend/main.py >> "$LOG_FILE" 2>&1 &
else
	if command -v conda >/dev/null 2>&1; then
		LAUNCH_MODE="conda"
		CMD_DESC="conda run -p $BACKEND_CONDA_ENV python backend/main.py"
		nohup conda run -p "$BACKEND_CONDA_ENV" python backend/main.py >> "$LOG_FILE" 2>&1 &
	else
		nohup python backend/main.py >> "$LOG_FILE" 2>&1 &
	fi
fi
PID=$!

{
	echo ""
	echo "[$START_TS] backend launch requested"
	echo "[$START_TS] env file: $ENV_FILE ($ENV_STATUS)"
	echo "[$START_TS] root dir: $ROOT_DIR"
	echo "[$START_TS] launch mode: $LAUNCH_MODE"
	echo "[$START_TS] command: $CMD_DESC"
	echo "[$START_TS] candidate pid: $PID"
} >> "$LOG_FILE"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
	{
		echo "[$(date '+%F %T')] backend failed shortly after launch; pid not alive"
	} >> "$LOG_FILE"
	echo "Backend failed to stay running. Check log: $LOG_FILE"
	exit 1
fi

echo "$PID" > "$PID_FILE"

{
	echo "[$(date '+%F %T')] backend started with PID=$PID"
	echo "[$(date '+%F %T')] log file: $LOG_FILE"
	echo "[$(date '+%F %T')] pid file: $PID_FILE"
} >> "$LOG_FILE"

echo "Backend started in background. PID=$PID"
echo "Log: $LOG_FILE"