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
STOP_TS="$(date '+%F %T')"

mkdir -p "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
    ENV_STATUS="loaded"
else
    ENV_STATUS="missing"
fi

if [[ ! -f "$PID_FILE" ]]; then
    {
        echo ""
        echo "[$STOP_TS] frontend stop requested"
        echo "[$STOP_TS] env file: $ENV_FILE ($ENV_STATUS)"
        echo "[$STOP_TS] root dir: $ROOT_DIR"
        echo "[$STOP_TS] result: pid file missing ($PID_FILE)"
    } >> "$LOG_FILE"
    echo "Frontend PID file not found: $PID_FILE"
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
    {
        echo ""
        echo "[$STOP_TS] frontend stop requested"
        echo "[$STOP_TS] result: pid file empty ($PID_FILE)"
    } >> "$LOG_FILE"
    echo "Frontend PID file is empty: $PID_FILE"
    rm -f "$PID_FILE"
    exit 0
fi

{
    echo ""
    echo "[$STOP_TS] frontend stop requested"
    echo "[$STOP_TS] env file: $ENV_FILE ($ENV_STATUS)"
    echo "[$STOP_TS] root dir: $ROOT_DIR"
    echo "[$STOP_TS] target pid: $PID"
} >> "$LOG_FILE"

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    for _ in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        {
            echo "[$(date '+%F %T')] frontend stop: grace period exceeded, sending SIGKILL to PID=$PID"
        } >> "$LOG_FILE"
        echo "Frontend PID=$PID did not exit in grace period, sending SIGKILL"
        kill -9 "$PID"
    fi
    {
        echo "[$(date '+%F %T')] frontend stopped. pid=$PID"
    } >> "$LOG_FILE"
    echo "Frontend stopped. PID=$PID"
else
    {
        echo "[$(date '+%F %T')] frontend stop: process not running for pid=$PID"
    } >> "$LOG_FILE"
    echo "Frontend process not running. PID=$PID"
fi

rm -f "$PID_FILE"
{
    echo "[$(date '+%F %T')] frontend pid file removed: $PID_FILE"
} >> "$LOG_FILE"
