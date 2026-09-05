"""Small Ollama-compatible HTTP service backed by Qwen2.5-7B-Instruct."""

from __future__ import annotations

import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    "/root/autodl-tmp/projects/llm-beginner/task-4-rag/models/Qwen2.5-7B-Instruct",
)
MODEL_NAME = os.getenv("RAG_MODEL", "qwen2.5:7b-instruct")

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=quantization,
).eval()
generation_lock = threading.Lock()
app = FastAPI(title="Task 4 Qwen service")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    options: dict = Field(default_factory=dict)


@app.get("/api/tags")
def tags():
    return {"models": [{"name": MODEL_NAME, "model": MODEL_NAME}]}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not enabled")
    messages = [message.model_dump() for message in request.messages]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    options = request.options
    temperature = float(options.get("temperature", 0.2))
    max_new_tokens = int(options.get("num_predict", options.get("max_new_tokens", 512)))
    generation = {
        "max_new_tokens": max(1, min(max_new_tokens, 1024)),
        "repetition_penalty": float(options.get("repetition_penalty", 1.05)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation.update(
            do_sample=True,
            temperature=temperature,
            top_p=float(options.get("top_p", 0.9)),
        )
    else:
        generation["do_sample"] = False
    started = time.time()
    with generation_lock, torch.inference_mode():
        output = model.generate(**inputs, **generation)
    generated = output[0, inputs.input_ids.shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "model": MODEL_NAME,
        "created_at": int(time.time()),
        "message": {"role": "assistant", "content": answer},
        "done": True,
        "total_duration": time.time() - started,
    }
