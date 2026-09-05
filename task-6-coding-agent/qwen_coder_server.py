"""Ollama-compatible local API backed by the Qwen2.5-Coder Q4_K_M GGUF."""

from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv(
    "CODER_MODEL_PATH",
    "/root/autodl-tmp/ollama-models/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463",
)
MODEL_NAME = os.getenv("CODER_MODEL", "qwen2.5-coder:7b-instruct")
THREADS = max(1, int(os.getenv("CODER_THREADS", str(min(8, os.cpu_count() or 4)))))

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=int(os.getenv("CODER_CONTEXT", "8192")),
    n_batch=256,
    n_threads=THREADS,
    n_threads_batch=THREADS,
    n_gpu_layers=int(os.getenv("CODER_GPU_LAYERS", "0")),
    use_mmap=True,
    verbose=False,
)
generation_lock = threading.Lock()
app = FastAPI(title="Task 6 Qwen2.5-Coder service")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    options: dict = Field(default_factory=dict)


@app.get("/api/tags")
def tags() -> dict:
    return {
        "models": [
            {
                "name": MODEL_NAME,
                "model": MODEL_NAME,
                "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
            }
        ]
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not enabled")
    options = request.options
    started = time.time()
    with generation_lock:
        result = llm.create_chat_completion(
            messages=[message.model_dump() for message in request.messages],
            max_tokens=max(1, min(int(options.get("num_predict", 512)), 900)),
            temperature=float(options.get("temperature", 0.05)),
            top_p=float(options.get("top_p", 0.9)),
            repeat_penalty=float(options.get("repeat_penalty", 1.05)),
        )
    content = result["choices"][0]["message"]["content"] or ""
    return {
        "model": MODEL_NAME,
        "created_at": int(time.time()),
        "message": {"role": "assistant", "content": content.strip()},
        "done": True,
        "total_duration": time.time() - started,
    }
