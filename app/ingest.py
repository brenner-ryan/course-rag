"""Build the vector index. Run once before starting the server:

    python -m app.ingest
"""
from . import store

if __name__ == "__main__":
    n = store.build_index()
    print("indexed %d chunks" % n)
