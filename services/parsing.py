"""HTML → structured blocks.

`DocumentParser` takes a raw HTML page (MkDocs Material layout, e.g. the
FastAPI docs) and returns a flat, ordered list of `DocumentBlock`s. Each block
already carries its `heading_path` (the chain of headings above it), so the
downstream `ChunkingService` can do a purely linear pass.

The parser is pure and offline: HTML string in, dataclasses out. Fetching the
page is somebody else's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, Tag

BlockType = Literal["heading", "paragraph", "code", "list", "table", "quote"]

# MkDocs Material renders the page body into this element.
_CONTENT_SELECTOR = "article.md-content__inner"
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentBlock:
    type: BlockType
    text: str
    heading_path: tuple[str, ...]
    level: int | None = None       # headings only: 1-6
    language: str | None = None    # code only: "python", ... (often unknown)


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    blocks: list[DocumentBlock]


class DocumentParseError(Exception):
    pass


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> tuple[str, ...]:
    """Add a heading, dropping any sibling/deeper headings first.

    stack holds (level, text) pairs. A new h2 removes any h2..h6 that were open,
    then appends itself, so the stack always reflects the path to "here".
    """
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))
    return _current_path(stack)


def _current_path(stack: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(text for _, text in stack)


def _normalize_prose(text: str) -> str:
    """Collapse runs of whitespace/newlines to single spaces. Prose only."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _strip_headerlink(tag: Tag) -> None:
    """Remove the trailing "¶" permalink anchor MkDocs adds to every heading."""
    link = tag.find("a", class_="headerlink")
    if link is not None:
        link.extract()


def _heading_text(tag: Tag) -> str:
    _strip_headerlink(tag)
    return _normalize_prose(tag.get_text())


def _code_text(element: Tag) -> str:
    """Text of a code block, whitespace preserved (never normalized)."""
    code = element.find("code") or element.find("pre") or element
    return code.get_text().strip("\n")


def _detect_language(element: Tag) -> str | None:
    """Best effort. MkDocs sometimes puts `language-xxx` on a wrapper class."""
    for el in (element, element.find("code"), element.find("pre")):
        if el is None:
            continue
        for cls in el.get("class") or []:
            if cls.startswith("language-"):
                return cls.removeprefix("language-")
    return None


def _render_list(list_el: Tag, depth: int = 0) -> str:
    """Nested <ul>/<ol> → indented "- item" lines."""
    lines: list[str] = []
    prefix = "  " * depth + "- "
    for li in list_el.find_all("li", recursive=False):
        nested = li.find_all(["ul", "ol"], recursive=False)
        for n in nested:
            n.extract()  # remove so get_text() is just this item's own text
        text = _normalize_prose(li.get_text())
        if text:
            lines.append(prefix + text)
        for n in nested:
            lines.append(_render_list(n, depth + 1))
    return "\n".join(lines)


def _render_table(table_el: Tag) -> str:
    """<table> → "cell | cell | cell" lines, header row included."""
    rows: list[str] = []
    for tr in table_el.find_all("tr"):
        cells = [_normalize_prose(c.get_text()) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _emit(blocks: list[DocumentBlock], type_: BlockType, text: str,
          stack: list[tuple[int, str]], **extra: object) -> None:
    if text.strip():
        blocks.append(DocumentBlock(type_, text, _current_path(stack), **extra))  # type: ignore[arg-type]


def _walk(node: Tag, stack: list[tuple[int, str]], blocks: list[DocumentBlock]) -> None:
    """Depth-first over `node`'s children. Each child is either emitted as one
    block (and not descended into) or is a container we recurse through."""
    for child in node.children:
        if not isinstance(child, Tag):
            continue

        name = child.name
        classes = child.get("class") or []

        if name in _HEADING_TAGS:
            level = int(name[1])
            text = _heading_text(child)
            _push_heading(stack, level, text)
            _emit(blocks, "heading", text, stack, level=level)

        elif name == "p":
            if "admonition-title" in classes:
                continue  # the admonition wrapper handles its own title
            _emit(blocks, "paragraph", _normalize_prose(child.get_text()), stack)

        elif name == "pre":
            _emit(blocks, "code", _code_text(child), stack, language=_detect_language(child))

        elif name in ("ul", "ol"):
            _emit(blocks, "list", _render_list(child), stack)

        elif name == "table":
            _emit(blocks, "table", _render_table(child), stack)

        elif name == "blockquote":
            _emit(blocks, "quote", _normalize_prose(child.get_text()), stack)

        elif name == "div":
            if "termy" in classes:
                continue  # animated fake-terminal widget — unparseable markup, low value
            if "highlight" in classes:
                _emit(blocks, "code", _code_text(child), stack, language=_detect_language(child))
            elif "tabbed-set" in classes:
                # tabs show the same example N times — keep only the first
                first = child.select_one(".tabbed-content .tabbed-block")
                if first is not None:
                    _walk(first, stack, blocks)
            else:
                # admonition, or any generic wrapper div
                _walk(child, stack, blocks)

        # img, hr, nav, script, style, ... — ignored


def _extract_title(content: Tag, soup: BeautifulSoup) -> str:
    h1 = content.find("h1")
    if h1 is not None:
        return _heading_text(h1)
    if soup.title and soup.title.string:
        return soup.title.string.split(" - ")[0].strip()
    return "Untitled"


class DocumentParser:
    def parse(self, raw_html: str) -> ParsedDocument:
        soup = BeautifulSoup(raw_html, "lxml")
        content = soup.select_one(_CONTENT_SELECTOR)
        if content is None:
            raise DocumentParseError(f"content region {_CONTENT_SELECTOR!r} not found")

        title = _extract_title(content, soup)
        blocks: list[DocumentBlock] = []
        _walk(content, [], blocks)
        return ParsedDocument(title=title, blocks=blocks)
