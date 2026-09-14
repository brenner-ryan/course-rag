"""Generation. Two backends, and the default one adds no dependencies at all.

The local backend runs flan-t5-small as ONNX through onnxruntime. onnxruntime,
tokenizers, numpy and huggingface_hub are already installed because ChromaDB needs them
for embeddings, so the local language model costs nothing beyond ~95 MB of model files.

The obvious way to do this is transformers plus torch. That was the first implementation
and it worked, but torch and its dependencies are about 990 MB installed, purely to run a
model with 80 million parameters. Running the same model through the ONNX runtime that was
already present takes the whole project from roughly 1.9 GB to under 700 MB.

Generation is greedy, with no KV cache: the decoder re-runs over the whole sequence each
step. That is O(n^2) and on a model this small it is not noticeable. It is also about
thirty lines instead of a hundred, and the cache path is the part that goes subtly wrong.
"""
import json
import urllib.request

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from . import config

MAX_INPUT_TOKENS = 512
DECODER_START = 0          # T5 starts decoding from the pad token
EOS = 1

_enc = _dec = _tok = None


def _load():
    """Download (once, then cached) and open the ONNX sessions and tokenizer."""
    global _enc, _dec, _tok
    if _tok is None:
        repo = config.LOCAL_MODEL
        _tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        opts = ort.SessionOptions()
        # Pin the thread count. Left to itself onnxruntime tries to set thread affinity,
        # fails inside a container, and prints an alarming error that is not one.
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        v = config.ONNX_VARIANT
        _enc = ort.InferenceSession(
            hf_hub_download(repo, "onnx/encoder_model%s.onnx" % v),
            opts, providers=["CPUExecutionProvider"])
        _dec = ort.InferenceSession(
            hf_hub_download(repo, "onnx/decoder_model%s.onnx" % v),
            opts, providers=["CPUExecutionProvider"])
    return _enc, _dec, _tok


def count_tokens(text):
    if config.LLM_BACKEND == "openai":
        return 0
    _, _, tok = _load()
    return len(tok.encode(text).ids)


def fit_context(context, overhead):
    """Trim `context` so that context + overhead fits the model's input window.

    Measured in TOKENS, not characters. A character budget cannot bound a token limit:
    700 characters of prose is around 170 tokens, and 700 characters of a dense table
    can be several times that.

    A no-op on the openai backend, whose window is larger and unknown here.
    """
    if config.LLM_BACKEND == "openai":
        return context
    _, _, tok = _load()
    budget = MAX_INPUT_TOKENS - overhead - 8          # 8 tokens of slack
    ids = tok.encode(context).ids
    if len(ids) <= budget:
        return context
    return tok.decode(ids[:max(0, budget)], skip_special_tokens=True)


def _local(prompt):
    enc, dec, tok = _load()
    ids = tok.encode(prompt).ids[:MAX_INPUT_TOKENS]
    input_ids = np.array([ids], dtype=np.int64)
    attn = np.ones_like(input_ids)

    hidden = enc.run(None, {"input_ids": input_ids, "attention_mask": attn})[0]

    out = [DECODER_START]
    for _ in range(config.MAX_NEW_TOKENS):
        logits = dec.run(["logits"], {
            "encoder_attention_mask": attn,
            "input_ids": np.array([out], dtype=np.int64),
            "encoder_hidden_states": hidden,
        })[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        out.append(nxt)
    return tok.decode(out[1:], skip_special_tokens=True).strip()


def _openai(prompt):
    if not config.LLM_BASE_URL:
        raise RuntimeError("LLM_BACKEND=openai but LLM_BASE_URL is not set")
    body = json.dumps({
        "model": config.LLM_MODEL or "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.MAX_NEW_TOKENS,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + config.LLM_API_KEY},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def generate(prompt):
    if config.LLM_BACKEND == "openai":
        return _openai(prompt)
    return _local(prompt)
