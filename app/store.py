"""Vector store over the course corpus.

Uses ChromaDB with its DEFAULT embedding function (all-MiniLM-L6-v2, via onnxruntime).
That is a deliberate choice: it downloads itself on first use and needs no API key and
no reachable server, so the project works for anyone who clones it. An earlier design
called a remote embedding service, which would have made the repository unusable by
anybody outside that network.
"""
import chromadb

from . import config
from .data import iter_lessons


def chunk(text, size=None, overlap=None):
    """Split on character count with overlap, preferring to break at a sentence end.

    Overlap exists so a fact sitting across a boundary is still retrievable whole from
    at least one chunk.
    """
    size = size or config.CHUNK_CHARS
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            dot = text.rfind(". ", start + size // 2, end)
            if dot != -1:
                end = dot + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [c for c in out if c]


def get_client():
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def build_index(client=None, collection_name=None):
    """(Re)build the collection from the corpus. Returns the number of chunks stored.

    Deletes and rebuilds rather than upserting, so a rebuild after editing the corpus
    cannot leave orphaned chunks behind answering questions from deleted content.
    """
    client = client or get_client()
    name = collection_name or config.COLLECTION
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for course, lesson in iter_lessons():
        for i, piece in enumerate(chunk(lesson["content"])):
            ids.append("%s::L%d::%d" % (course["title"], lesson["number"], i))
            docs.append("%s, lesson %d: %s\n\n%s" %
                        (course["title"], lesson["number"], lesson["title"], piece))
            metas.append({
                "course": course["title"],
                "instructor": course["instructor"],
                "link": course["link"],
                "lesson_number": lesson["number"],
                "lesson_title": lesson["title"],
            })
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"})


def search(query, k=None, client=None):
    """Return up to k chunks most similar to the query, best first."""
    col = get_collection(client)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k or config.TOP_K, col.count()))
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "meta": meta, "distance": dist})
    return out
