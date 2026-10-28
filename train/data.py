"""The corpus: classic English novels from Project Gutenberg (public domain).

Downloads each book, strips the Gutenberg licence header/footer, un-wraps the
hard line breaks inside paragraphs, and removes typesetting noise
(_italics_ markers, [Illustration] lines, page numbers).
"""

from __future__ import annotations

import os
import re
import urllib.request

# (Gutenberg id, title, author)
BOOKS = [
    (1342, "Pride and Prejudice", "Jane Austen"),
    (158, "Emma", "Jane Austen"),
    (161, "Sense and Sensibility", "Jane Austen"),
    (105, "Persuasion", "Jane Austen"),
    (141, "Mansfield Park", "Jane Austen"),
    (121, "Northanger Abbey", "Jane Austen"),
    (1260, "Jane Eyre", "Charlotte Brontë"),
    (9182, "Villette", "Charlotte Brontë"),
    (768, "Wuthering Heights", "Emily Brontë"),
    (969, "The Tenant of Wildfell Hall", "Anne Brontë"),
    (1400, "Great Expectations", "Charles Dickens"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (730, "Oliver Twist", "Charles Dickens"),
    (766, "David Copperfield", "Charles Dickens"),
    (1023, "Bleak House", "Charles Dickens"),
    (84, "Frankenstein", "Mary Shelley"),
    (345, "Dracula", "Bram Stoker"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (145, "Middlemarch", "George Eliot"),
    (6688, "The Mill on the Floss", "George Eliot"),
    (110, "Tess of the d'Urbervilles", "Thomas Hardy"),
    (107, "Far from the Madding Crowd", "Thomas Hardy"),
    (2641, "A Room with a View", "E. M. Forster"),
    (4276, "North and South", "Elizabeth Gaskell"),
    (599, "Vanity Fair", "William Makepeace Thackeray"),
]

START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def clean(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    m = START.search(text)
    if m:
        text = text[m.end() :]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"\[(Illustration|Footnote)[^\]]*\]", "", text, flags=re.I | re.S)
    text = re.sub(r"_([^_\n]+(?:\n[^_\n]+)*)_", r"\1", text)  # _italics_
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.M)  # lone page numbers
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [re.sub(r" {2,}", " ", p) for p in paragraphs if p]
    return "\n".join(paragraphs).strip() + "\n"


def download(cache_dir: str) -> list[tuple[str, str]]:
    """Returns [(title, cleaned text)], caching downloads in `cache_dir`."""
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for gid, title, author in BOOKS:
        path = os.path.join(cache_dir, f"{gid}.txt")
        if not os.path.exists(path):
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "glance-training/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read().decode("utf-8", errors="replace")
            except Exception as e:  # one missing book shouldn't stop training
                print(f"  skipped {title}: {e}")
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean(raw))
        with open(path, encoding="utf-8") as f:
            text = f.read()
        out.append((f"{title} ({author})", text))
        print(f"  {title}: {len(text) / 1e6:.2f} M characters")
    return out
