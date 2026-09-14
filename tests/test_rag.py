"""Tests for the retrieval and API layers.

Generation is stubbed in most tests on purpose. The model is slow on a CPU and its output
is not deterministic, so asserting on it would make the suite slow and flaky while telling
us nothing about our own code. What we actually own, and therefore what is tested here, is
chunking, indexing, retrieval ranking, citation, session handling and the API contract.
One test does exercise the real model, marked slow, so the wiring is covered too.
"""
import os
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "2")

from fastapi.testclient import TestClient

from app import config, rag, store
from app.data import load_courses
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def index(tmp_path_factory):
    """Build a throwaway index so tests never touch a real one."""
    config.CHROMA_DIR = str(tmp_path_factory.mktemp("chroma"))
    n = store.build_index()
    assert n > 0
    return n


@pytest.fixture(autouse=True)
def clear_sessions():
    rag.SESSIONS.clear()


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(rag, "generate", lambda prompt: "STUB:" + str(len(prompt)))


# --- corpus parsing -------------------------------------------------------

def test_corpus_parses():
    cs = load_courses()
    assert len(cs) == 3
    for c in cs:
        assert c["title"] and c["instructor"] and c["link"].startswith("http")
        assert len(c["lessons"]) == 3
        for l in c["lessons"]:
            assert l["number"] in (1, 2, 3)
            assert l["title"] and len(l["content"]) > 200


def test_parser_handles_the_documented_shape():
    from app.data import parse_course
    c = parse_course(
        "# Title Here\n\ninstructor: A Person\nlink: https://example.edu/x\n\n"
        "## Lesson 1: First\n\nOne\nwrapped line.\n\nSecond para.\n\n"
        "## Lesson 2: Second\n\nBody.\n")
    assert c["title"] == "Title Here"
    assert c["instructor"] == "A Person"
    assert [l["number"] for l in c["lessons"]] == [1, 2]
    # wrapped source lines are rejoined, blank-line paragraph breaks are kept
    assert c["lessons"][0]["content"] == "One wrapped line.\n\nSecond para."


# --- chunking -------------------------------------------------------------

def test_short_text_is_one_chunk():
    assert store.chunk("hello world") == ["hello world"]


def test_long_text_splits_and_covers_everything():
    text = ". ".join("sentence number %d is here" % i for i in range(200)) + "."
    chunks = store.chunk(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    # every sentence survives somewhere
    for i in (0, 99, 199):
        assert any("sentence number %d " % i in c for c in chunks)


def test_chunking_always_terminates():
    # overlap >= size would loop forever without the guard in chunk()
    assert len(store.chunk("x" * 900, size=100, overlap=500)) > 0


# --- indexing and retrieval ----------------------------------------------

def test_every_lesson_is_indexed(index):
    lessons = sum(len(c["lessons"]) for c in load_courses())
    assert index >= lessons


def test_retrieval_finds_the_right_lesson():
    """Each query names something discussed in exactly one lesson."""
    cases = [
        ("why does bread get soggy", "Introduction to Sandwich Engineering", 2),
        ("can an intercepted message be altered", "Practical Pigeon Communication", 3),
        ("what room temperature is best", "Foundations of Competitive Napping", 2),
    ]
    for query, course, lesson in cases:
        hits = store.search(query, k=1)
        assert hits, query
        assert hits[0]["meta"]["course"] == course, query
        assert hits[0]["meta"]["lesson_number"] == lesson, query


def test_search_returns_at_most_k():
    assert len(store.search("sandwich", k=2)) <= 2


# --- answering ------------------------------------------------------------

def test_answer_cites_sources(stub):
    out = rag.answer("why does bread get soggy")
    assert out["sources"]
    assert out["sources"][0]["course"] == "Introduction to Sandwich Engineering"
    assert "lesson_title" in out["sources"][0]


def test_sources_are_deduplicated(stub):
    out = rag.answer("sandwich failure modes")
    seen = [(s["course"], s["lesson_number"]) for s in out["sources"]]
    assert len(seen) == len(set(seen))


def test_context_is_actually_passed_to_the_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(rag, "generate", lambda p: captured.setdefault("p", p) or "ok")
    rag.answer("why does bread get soggy")
    assert "sogginess" in captured["p"].lower()


# --- sessions -------------------------------------------------------------

def test_session_is_created_and_reused(stub):
    first = rag.answer("question one")
    second = rag.answer("question two", first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert len(rag.history(first["session_id"])) == 2


def test_history_is_capped(stub):
    sid = rag.new_session()
    for i in range(rag.MAX_TURNS + 4):
        rag.answer("q%d" % i, sid)
    assert len(rag.history(sid)) == rag.MAX_TURNS


def test_unknown_session_id_does_not_crash(stub):
    out = rag.answer("hello", "not-a-real-session")
    assert out["session_id"] != "not-a-real-session"


# --- API ------------------------------------------------------------------

def test_chat_endpoint(stub):
    c = TestClient(app)
    r = c.post("/api/chat", json={"question": "why does bread get soggy"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("STUB:")
    assert body["sources"] and body["session_id"]


def test_courses_endpoint():
    r = TestClient(app).get("/api/courses")
    assert r.status_code == 200
    assert len(r.json()) == len(load_courses())
    assert all(c["lessons"] for c in r.json())


def test_stats_endpoint():
    r = TestClient(app).get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["indexed_chunks"] > 0
    assert body["courses"] == len(load_courses())


def test_session_endpoint(stub):
    c = TestClient(app)
    sid = c.post("/api/chat", json={"question": "hello"}).json()["session_id"]
    r = c.get("/api/session/" + sid)
    assert r.status_code == 200
    assert len(r.json()["turns"]) == 1


def test_frontend_is_served():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "Course Materials RAG" in r.text


# --- prompt fits the model's input window ---------------------------------

def test_prompt_fits_the_input_window():
    """A real prompt exceeded flan-t5-small's 512-token input and was silently truncated.

    Because the question used to sit at the end of the template, the truncation was
    cutting off the question itself and the model answered from context alone. This
    asserts the assembled prompt fits, for a question against every course.
    """
    from app import llm

    questions = [
        "What are the three sandwich failure modes and how do they differ?",
        "Is a message carried by pigeon authenticated in any way at all?",
        "What ambient temperature is recommended for competitive napping?",
    ]
    for q in questions:
        hits = store.search(q)
        ctx = "\n\n---\n\n".join(h["text"] for h in hits)
        overhead = llm.count_tokens(rag.PROMPT.format(context="", question=q))
        ctx = llm.fit_context(ctx, overhead)
        n = llm.count_tokens(rag.PROMPT.format(context=ctx, question=q))
        assert n <= llm.MAX_INPUT_TOKENS, "%d tokens for %r" % (n, q)


def test_question_precedes_context_in_the_prompt():
    """Ordering is load-bearing. If anything is dropped it must not be the question."""
    built = rag.PROMPT.format(context="CONTEXT_MARKER", question="QUESTION_MARKER")
    assert built.index("QUESTION_MARKER") < built.index("CONTEXT_MARKER")


# --- the real model -------------------------------------------------------

@pytest.mark.slow
def test_real_model_answers_from_context():
    """Not asserting the answer is good. Asserting the whole chain runs and returns text."""
    out = rag.answer("What are the three sandwich failure modes?")
    assert isinstance(out["answer"], str) and out["answer"].strip()
    assert out["sources"][0]["course"] == "Introduction to Sandwich Engineering"
