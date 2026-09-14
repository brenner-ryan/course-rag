# Architecture

688 lines across nine files. Two pipelines: one that runs once to build the index, and one
that runs per question.

| File | Lines | Responsibility |
|---|---|---|
| `app/config.py` | 30 | every setting, as an environment variable with a working default |
| `app/data.py` | 22 | loads the corpus, yields (course, lesson) pairs |
| `app/store.py` | 89 | chunking, index construction, retrieval |
| `app/llm.py` | 67 | generation: local model, or any OpenAI-compatible endpoint |
| `app/rag.py` | 70 | the orchestrator: session, retrieve, build context, generate, cite |
| `app/main.py` | 99 | FastAPI routes, request and response schemas, static serving |
| `app/ingest.py` | 9 | entry point for building the index |
| `static/index.html` | 126 | the entire frontend, no build step |
| `tests/test_rag.py` | 176 | 18 tests |

## Pipeline 1: ingest, run once

```
  data/courses.json                         3 courses, 9 lessons, 5,827 characters
        |
        v
  data.iter_lessons()                       yields (course, lesson) pairs
        |
        v
  store.chunk()                             800 chars, 100 overlap,
        |                                   prefers to break at a sentence end
        v
  store.build_index()                       id   = "<course>::L<n>::<i>"
        |                                   doc  = "<course>, lesson <n>: <title>\n\n<text>"
        |                                   meta = course, instructor, link,
        |                                          lesson_number, lesson_title
        v
  ChromaDB .add()
        |
        v
  all-MiniLM-L6-v2                          384-dim vector per chunk
  (onnxruntime, inside Chroma)              <- our code never sees a vector
        |
        v
  ./chroma_db/                              persistent, HNSW index, cosine space
```

The collection is **deleted and rebuilt**, not upserted. An upsert after editing the corpus
would leave orphaned chunks behind, still answering questions from content that no longer
exists.

## Pipeline 2: one question

```
  browser  static/index.html  ask()
        |   POST /api/chat  {question, session_id}
        v
  main.py  chat()                           pydantic ChatRequest validates the body
        |
        v
  rag.answer(question, session_id)
        |
        +--1--> session                     SESSIONS[sid], or a new uuid4 if the id is
        |                                   missing or unknown
        |
        +--2--> store.search(question)
        |            |
        |            v
        |       ChromaDB col.query(n_results=3)
        |            |
        |            v
        |       all-MiniLM-L6-v2            embeds the QUERY this time
        |            |
        |            v
        |       HNSW cosine search          over the stored chunks
        |            |
        |            v
        |       top 3 {text, meta, distance}
        |
        +--3--> context                     the chunk texts joined by "---"
        |
        +--4--> llm.generate(prompt)        prompt = instruction + context + question
        |            |
        |            +-- LLM_BACKEND=local  --> flan-t5-small on the CPU   [default]
        |            +-- LLM_BACKEND=openai --> POST {LLM_BASE_URL}/chat/completions
        |
        +--5--> sources                     hits deduplicated by (course, lesson_number)
        |
        +--6--> history                     append the turn, keep the last 6
                     |
                     v
        {answer, sources, session_id}
                     |
                     v
  browser renders the answer and the lesson links
```

## Things worth knowing

**Our code never touches a vector.** Embedding happens inside ChromaDB, on both the write
path and the read path, using its default model. We pass text in and get text back. That is
why swapping the language model is a one-line environment variable while swapping the
embedding model would mean changing `store.py`.

**Retrieval is unconditional.** Every question triggers a search. The course's version gives
the model a search tool and lets it decide whether to call it, which is a trained behaviour
that an 80M parameter model does not perform reliably. The trade is discussed in
`REFLECTION.md`.

**The same model embeds queries and documents.** all-MiniLM-L6-v2 is symmetric, so no task
prefix is needed. Some embedding models are not, and using the wrong prefix measurably
degrades retrieval.

**Sessions are in memory and die with the process.** A dict keyed by uuid, capped at six
turns. Adequate for a demo and honestly wrong for anything real, where it would need a store
of its own.

**History is kept but not yet used in the prompt.** `rag.answer` records each turn and the
`/api/session/{id}` endpoint returns them, but the prompt sent to the model contains only
the retrieved context and the current question. Follow-up questions that depend on the
previous answer will therefore not resolve. This is the most obvious next feature.

**Failure is degraded, not fatal.** An empty index returns a message telling you to run the
ingest rather than raising. A question with no good match still returns whatever came back,
and the prompt instructs the model to say it does not know.
