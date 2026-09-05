#!/usr/bin/env bash
set -euo pipefail

TASK_DIR=$(cd "$(dirname "$0")" && pwd)
source "$TASK_DIR/qwen.env"

if ! curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  screen -S qwen-service -X quit >/dev/null 2>&1 || true
  screen -L -Logfile "$TASK_DIR/qwen_service.log" -dmS qwen-service \
    env QWEN_MODEL_PATH="$QWEN_MODEL_PATH" \
    /root/miniconda3/envs/llm/bin/python -m uvicorn qwen_server:app \
    --app-dir "$TASK_DIR" --host "$QWEN_HOST" --port "$QWEN_PORT"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null
echo "Qwen service ready: $RAG_API_URL ($RAG_MODEL)"
