"""The corpus, and the parser for it. Entirely synthetic.

Each course is one Markdown file in data/courses/, so that a reader can open the corpus
and actually read it. An earlier version held the same text in a single JSON file, where
each lesson was one 900-character line and nobody could see what the courses said.

The format is deliberately plain:

    # Course Title

    instructor: Some Name
    link: https://example.edu/...

    ## Lesson 1: Lesson Title

    Body paragraphs.

    ## Lesson 2: Another Title

    More body.

No real course material is used. This repository is public, and real coursework is
somebody's education record.
"""
import re
from pathlib import Path

COURSE_DIR = Path(__file__).parent.parent / "data" / "courses"


def parse_course(text):
    title = ""
    meta = {}
    lessons = []
    current = None
    body = []

    def flush():
        if current is not None:
            current["content"] = "\n\n".join(
                " ".join(p.split()) for p in "\n".join(body).strip().split("\n\n") if p.strip()
            )
            lessons.append(current)

    for line in text.splitlines():
        heading = re.match(r"^##\s+Lesson\s+(\d+)\s*:\s*(.+)$", line)
        if heading:
            flush()
            body = []
            current = {"number": int(heading.group(1)), "title": heading.group(2).strip()}
            continue
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if current is None:
            kv = re.match(r"^(\w+)\s*:\s*(.+)$", line.strip())
            if kv:
                meta[kv.group(1).lower()] = kv.group(2).strip()
            continue
        body.append(line)
    flush()

    return {
        "title": title,
        "instructor": meta.get("instructor", ""),
        "link": meta.get("link", ""),
        "lessons": lessons,
    }


def load_courses():
    """Every course, sorted by title so ordering is stable across filesystems."""
    courses = [parse_course(p.read_text()) for p in sorted(COURSE_DIR.glob("*.md"))]
    return sorted(courses, key=lambda c: c["title"])


def iter_lessons():
    """Yield (course, lesson) pairs across every course."""
    for course in load_courses():
        for lesson in course["lessons"]:
            yield course, lesson
