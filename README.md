# Course Materials RAG

A retrieval-augmented chatbot over course materials. Ask a question, get an answer built
only from the indexed lessons, with the lessons it used shown underneath.

## What it does

- Chunks and indexes a corpus of courses and lessons into a local vector database
- Retrieves the most relevant chunks for a question by semantic similarity
- Generates an answer from those chunks with a small local language model
- Cites which course and lesson each answer came from
- Keeps per-session conversation history
- Exposes corpus and index statistics, and interactive API docs

## It runs on your machine, with no account and no key

There is no hosted service behind this and nothing to sign up for. The embedding model and
the large language model both download themselves on first run and then run locally,
offline, on the machine that cloned the repository.

## Requirements

Python 3.11 or newer. No GPU, no API key, no account.

About 1 GB of disk: 459 MB of Python packages, and two models totalling 543 MB that
download themselves on first use. Setting `ONNX_VARIANT=_quantized` brings the language
model down from 376 MB to 97 MB, at a real cost in answer quality described below.

There is **no machine-learning dependency in `requirements.txt`**. The language model runs
on `onnxruntime`, which arrives as a ChromaDB dependency along with `tokenizers`, `numpy`
and `huggingface_hub`, because Chroma uses all four for embeddings. An earlier version used
`transformers` and `torch`, which added about 990 MB to run a model with 80 million
parameters.

## Setup

```bash
git clone <this repo>
cd course-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

```bash
python -m app.ingest        # build the vector index (downloads the embedding model, ~80 MB)
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000>. Interactive API documentation is at
<http://127.0.0.1:8000/docs>.

The first question downloads the language model, so it takes a while. After that answers
take roughly one to six seconds on a CPU.

## About the answers

They are not clever. The default model is `flan-t5-small`, which has 80 million parameters
and is a few years old, run as ONNX. It was chosen because it downloads in seconds, runs on
any CPU, and needs no key, which makes this repository something a reader can actually run.
Answer quality is not what this project is demonstrating; retrieval, citation, session
handling and the surrounding plumbing are.

If you want better answers, point it at any OpenAI-compatible endpoint:

```bash
export LLM_BACKEND=openai
export LLM_BASE_URL=http://localhost:11434/v1   # Ollama, llama.cpp, LM Studio, a hosted API
export LLM_MODEL=qwen2.5:7b
```

## Configuration

Every setting has a working default. All are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `local` | `local` or `openai` |
| `LOCAL_MODEL` | `Xenova/flan-t5-small` | ONNX model repository used when backend is `local` |
| `ONNX_VARIANT` | *(empty)* | `_quantized` for a 97 MB int8 graph instead of 376 MB, at a real quality cost |
| `LLM_BASE_URL` | *(empty)* | OpenAI-compatible base URL, required when backend is `openai` |
| `LLM_MODEL` | *(empty)* | model name to request from that endpoint |
| `CHROMA_DIR` | `./chroma_db` | where the vector index is stored |
| `CHUNK_CHARS` | `800` | characters per chunk |
| `CHUNK_OVERLAP` | `100` | overlap between chunks |
| `TOP_K` | `3` | chunks retrieved per question |
| `MAX_NEW_TOKENS` | `160` | answer length cap |

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | ask a question; returns answer, sources, session id |
| `GET /api/courses` | list indexed courses and lessons |
| `GET /api/stats` | corpus and index statistics |
| `GET /api/session/{id}` | conversation history for a session |
| `GET /docs` | interactive API documentation |

## The data is invented

`data/courses/` contains three fictional courses, one Markdown file each: sandwich
engineering, pigeon communication, and competitive napping. Open them and read them. None
of it is real course material. This repository is public, and real coursework is somebody's
education record.

Adding a course means dropping another Markdown file in that directory and re-running
`python -m app.ingest`. The format is a `#` title, `instructor:` and `link:` lines, then
`## Lesson N: Title` sections.

The content is deliberately absurd but internally consistent, so retrieval can be tested
properly: each lesson contains facts that appear in no other lesson, which means a query
has exactly one correct answer and a wrong retrieval is obvious.

## Tests

```bash
pytest -m "not slow"   # 17 tests, about 15 seconds
pytest                 # adds one test that runs the real model, about 35 seconds
```

The suite covers chunking, index construction, retrieval ranking, source citation and
deduplication, session creation and history capping, and every API endpoint.

Generation is stubbed in most tests on purpose. The model is slow and its output is not
deterministic, so asserting on the text would make the suite slow and flaky while testing
somebody else's model rather than this code. One test does drive the real model end to end,
marked `slow`, so the wiring is covered.

## Layout

```
app/config.py    settings, all environment variables with working defaults
app/data.py      loads the corpus
app/store.py     chunking, indexing, retrieval
app/llm.py       generation, local or OpenAI-compatible
app/rag.py       retrieve, stuff, answer, plus session history
app/ingest.py    build the index
app/main.py      FastAPI application
static/          single-page frontend, no build step
tests/           pytest suite
```

## Differences from the course's version

Two, both deliberate.

The course application gives the model a search **tool** and lets it decide whether and when
to call it. This one always retrieves and then hands the results to the model. Deciding to
call a function and emitting well-formed arguments is a trained behaviour that an 80M
parameter model does not perform reliably.

The course used Anthropic's API for generation. This uses a local model by default, so the
project runs for anyone who clones it rather than only for someone holding a key.

## A note on the quantized graph

`ONNX_VARIANT=_quantized` is tempting: 97 MB instead of 376 MB for the same model. On a
model this small the damage is not subtle. Asked for three failure modes it returns two,
and asked a yes or no question it answered `ii.`. Full precision is the default for that
reason, and the switch is documented rather than removed because on a larger model the
trade would look different.
