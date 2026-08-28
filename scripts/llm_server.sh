#!/usr/bin/env bash
# ==============================================================================
# Dedicated LLM Server Launcher Script (FreeToken on GPU 1)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env if present
if [[ -f "${ROOT_DIR}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
fi

FREETOKEN_DIR="/home/aiserver/LABS/FREETOKEN/FreeToken"
FT_VENV_BIN="${FREETOKEN_DIR}/.venv/bin/ft"
PORT="${LLM_PORT:-4050}"
HOST="${LLM_HOST:-0.0.0.0}"
MODEL="${LLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/llm_server.log"
PID_FILE="${LOG_DIR}/llm_server.pid"

mkdir -p "${LOG_DIR}"

is_running() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    fi
    if lsof -i ":${PORT}" -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

start_server() {
    if is_running; then
        echo "==> LLM server is already running on port ${PORT}."
        status_server
        return 0
    fi

    if [[ ! -x "${FT_VENV_BIN}" ]]; then
        echo "==> Error: FreeToken binary not found at ${FT_VENV_BIN}"
        exit 1
    fi

    echo "==> Starting FreeToken server..."
    echo "    Model: ${MODEL}"
    echo "    Port:  ${PORT}"
    echo "    GPU:   ${GPU_ID}"
    echo "    Log:   ${LOG_FILE}"

    # Free up port if stale
    fuser -k "${PORT}/tcp" "${PORT}+1/tcp" >/dev/null 2>&1 || true
    sleep 1

    CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${FT_VENV_BIN}" serve \
        --model "${MODEL}" \
        --port "${PORT}" \
        --host "${HOST}" \
        --cors-origins '*' \
        > "${LOG_FILE}" 2>&1 &

    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    echo "==> FreeToken started in background with PID: ${pid}"

    echo -n "==> Waiting for server to initialize..."
    local attempts=0
    local max_attempts=45
    while [[ ${attempts} -lt ${max_attempts} ]]; do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            local health
            health="$(curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)"
            if echo "${health}" | grep -q '"status":"ok"'; then
                echo " [READY]"
                echo "==> Server is active and ready: http://127.0.0.1:${PORT}/v1"
                return 0
            else
                echo -n "."
            fi
        else
            echo -n "."
        fi
        sleep 2
        attempts=$((attempts + 1))
    done

    echo " [INITIALIZING IN BACKGROUND]"
    echo "==> Model is still loading expert weights. Tail logs with:"
    echo "    ${0} logs"
}

stop_server() {
    echo "==> Stopping LLM server on port ${PORT}..."
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 1
            kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
    fi

    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    echo "==> Server stopped."
}

status_server() {
    if is_running; then
        echo "● LLM server is RUNNING on port ${PORT}"
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            local health
            health="$(curl -s "http://127.0.0.1:${PORT}/health")"
            echo "  Health: ${health}"
        else
            echo "  Health: responding on socket (initializing...)"
        fi
    else
        echo "○ LLM server is STOPPED"
    fi
}

logs_server() {
    if [[ -f "${LOG_FILE}" ]]; then
        tail -f "${LOG_FILE}"
    else
        echo "No log file found at ${LOG_FILE}"
    fi
}

case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        status_server
        ;;
    logs)
        logs_server
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
