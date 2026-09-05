#!/usr/bin/env bash
set -euo pipefail

TASK_DIR=$(cd "$(dirname "$0")" && pwd)
HOST=${CODER_HOST:-127.0.0.1}
PORT=${CODER_PORT:-11435}
MODEL=${CODER_MODEL:-qwen2.5-coder:7b-instruct}

if ! curl -fsS "http://$HOST:$PORT/api/tags" >/dev/null 2>&1; then
  screen -S qwen-coder -X quit >/dev/null 2>&1 || true
  screen -L -Logfile "$TASK_DIR/qwen_coder.log" -dmS qwen-coder \
    env CODER_MODEL="$MODEL" \
    /root/miniconda3/envs/llm/bin/python -m uvicorn qwen_coder_server:app \
    --app-dir "$TASK_DIR" --host "$HOST" --port "$PORT"
  for _ in $(seq 1 180); do
    curl -fsS "http://$HOST:$PORT/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl -fsS "http://$HOST:$PORT/api/tags"
echo
echo "Qwen Coder ready: http://$HOST:$PORT/api/chat ($MODEL)"
