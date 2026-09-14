"""Retrieve, stuff, answer. Plus per-session conversation history.

This is deliberately the simple form of RAG: the retrieval always happens, and the model
is handed the results. The course's own application instead gives the model a search tool
and lets it decide whether to call it. That is a trained behaviour which a model this
small does not perform reliably, so the decision is made in code instead. The trade is
recorded in the README rather than hidden.
"""
import uuid

from . import store
from .llm import generate

SESSIONS = {}
MAX_TURNS = 6

PROMPT = """Answer the question using only the course material below.
If the material does not contain the answer, say you do not know.

Course material:
{context}

Question: {question}
Answer:"""


def new_session():
    sid = uuid.uuid4().hex
    SESSIONS[sid] = []
    return sid


def history(session_id):
    return SESSIONS.get(session_id, [])


def answer(question, session_id=None):
    """Return {answer, sources, session_id}. Creates a session if none is given."""
    if not session_id or session_id not in SESSIONS:
        session_id = new_session()

    hits = store.search(question)
    if not hits:
        reply = ("The course material has not been indexed yet. "
                 "Run `python -m app.ingest` and try again.")
        SESSIONS[session_id].append({"question": question, "answer": reply})
        return {"answer": reply, "sources": [], "session_id": session_id}

    context = "\n\n---\n\n".join(h["text"] for h in hits)
    reply = generate(PROMPT.format(context=context, question=question))

    sources = []
    seen = set()
    for h in hits:
        m = h["meta"]
        key = (m["course"], m["lesson_number"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "course": m["course"],
            "lesson_number": m["lesson_number"],
            "lesson_title": m["lesson_title"],
            "link": m["link"],
        })

    turns = SESSIONS[session_id]
    turns.append({"question": question, "answer": reply})
    del turns[:-MAX_TURNS]
    return {"answer": reply, "sources": sources, "session_id": session_id}
