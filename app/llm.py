"""Generation. Two backends, one of which needs nothing installed beyond the requirements.

The local backend runs google/flan-t5-small on the CPU. It is a small, old, and frankly
not very clever model, chosen because it downloads in seconds, runs anywhere, needs no
key, and answers well enough from provided context to demonstrate retrieval-augmented
generation. Answer quality is not what this project is demonstrating.

The openai backend exists so the same code can be pointed at a better model when one is
available, without the default path depending on hardware the reader does not have.
"""
import json
import urllib.request

from . import config

_model = None
_tok = None


def _load():
    """Load the seq2seq model directly rather than through a pipeline.

    transformers 5 removed the "text2text-generation" pipeline task, so the pipeline
    call that works on 4.x raises KeyError on 5.x. Driving the model and tokenizer
    directly works on both and does not depend on the pipeline registry at all.
    """
    global _model, _tok
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(config.LOCAL_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.LOCAL_MODEL)
        _model.eval()
    return _model, _tok


def _local(prompt):
    import torch
    model, tok = _load()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
    return tok.decode(out[0], skip_special_tokens=True).strip()


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
