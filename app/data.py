"""The corpus. Entirely synthetic.

Shaped like the course app's data (a course has a title, an instructor, a link and
numbered lessons) so the retrieval and citation paths are exercised the same way.
No real course material is used: this repository is public, and real coursework is
somebody's education record.
"""
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "courses.json"


def load_courses():
    return json.loads(DATA_FILE.read_text())


def iter_lessons():
    """Yield (course, lesson) pairs across every course."""
    for course in load_courses():
        for lesson in course["lessons"]:
            yield course, lesson
