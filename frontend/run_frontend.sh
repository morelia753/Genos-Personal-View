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
LOG_DIR="$ROOT_DIR/frontend/logs"
LOG_FILE="$LOG_DIR/frontend.nohup.log"
PID_FILE="$LOG_DIR/frontend.pid"

# Priority: explicit python path > conda env name > system python.
FRONTEND_PYTHON_BIN="${FRONTEND_PYTHON_BIN:-}"
FRONTEND_CONDA_ENV="${FRONTEND_CONDA_ENV:-gradio}"

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
			echo "[$START_TS] frontend launch skipped: already running"
			echo "[$START_TS] existing pid: $OLD_PID"
			echo "[$START_TS] log file: $LOG_FILE"
		} >> "$LOG_FILE"
		echo "Frontend is already running with PID=$OLD_PID"
		echo "Log: $LOG_FILE"
		exit 0
	fi
fi

LAUNCH_MODE="system-python"
CMD_DESC="python frontend/app.py"
if [[ -n "$FRONTEND_PYTHON_BIN" ]]; then
	LAUNCH_MODE="python-bin"
	CMD_DESC="$FRONTEND_PYTHON_BIN frontend/app.py"
	nohup "$FRONTEND_PYTHON_BIN" frontend/app.py >> "$LOG_FILE" 2>&1 &
else
	if command -v conda >/dev/null 2>&1; then
		LAUNCH_MODE="conda"
		CMD_DESC="conda run -p $FRONTEND_CONDA_ENV python frontend/app.py"
		nohup conda run -p "$FRONTEND_CONDA_ENV" python frontend/app.py >> "$LOG_FILE" 2>&1 &
	else
		nohup python frontend/app.py >> "$LOG_FILE" 2>&1 &
	fi
fi

PID=$!

{
	echo ""
	echo "[$START_TS] frontend launch requested"
	echo "[$START_TS] env file: $ENV_FILE ($ENV_STATUS)"
	echo "[$START_TS] root dir: $ROOT_DIR"
	echo "[$START_TS] launch mode: $LAUNCH_MODE"
	echo "[$START_TS] command: $CMD_DESC"
	echo "[$START_TS] candidate pid: $PID"
} >> "$LOG_FILE"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
	{
		echo "[$(date '+%F %T')] frontend failed shortly after launch; pid not alive"
	} >> "$LOG_FILE"
	echo "Frontend failed to stay running. Check log: $LOG_FILE"
	exit 1
fi

echo "$PID" > "$PID_FILE"

{
	echo "[$(date '+%F %T')] frontend started with PID=$PID"
	echo "[$(date '+%F %T')] log file: $LOG_FILE"
	echo "[$(date '+%F %T')] pid file: $PID_FILE"
} >> "$LOG_FILE"

echo "Frontend started in background. PID=$PID"
echo "Log: $LOG_FILE"