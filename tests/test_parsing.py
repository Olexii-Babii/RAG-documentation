import pathlib

import pytest

from services.parsing import DocumentBlock, DocumentParser, _push_heading

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_push_heading_builds_path():
    stack: list[tuple[int, str]] = []
    assert _push_heading(stack, 1, "FastAPI") == ("FastAPI",)
    assert _push_heading(stack, 2, "Tutorial") == ("FastAPI", "Tutorial")
    assert _push_heading(stack, 3, "Query Params") == ("FastAPI", "Tutorial", "Query Params")


def test_push_heading_pops_siblings_and_deeper():
    stack = [(1, "FastAPI"), (2, "Tutorial"), (3, "Query Params")]
    # a new h2 replaces the open h2 and drops the h3 under it
    assert _push_heading(stack, 2, "Advanced") == ("FastAPI", "Advanced")


def test_push_heading_skipped_levels():
    stack: list[tuple[int, str]] = []
    _push_heading(stack, 1, "A")
    assert _push_heading(stack, 3, "B") == ("A", "B")  # h2 skipped, still nests


_HTML = """
<html><body>
  <nav>Skip to content</nav>
  <article class="md-content__inner">
    <h1>Page<a class="headerlink" href="#">¶</a></h1>
    <p>Intro paragraph.</p>
    <h2>Section</h2>
    <p>Body   with\n  messy   whitespace.</p>
    <div class="highlight"><pre><code>x = 1\n\ny = 2\n</code></pre></div>
    <div class="admonition tip">
      <p class="admonition-title">Tip</p>
      <p>Useful hint.</p>
    </div>
    <ul><li>one</li><li>two<ul><li>nested</li></ul></li></ul>
  </article>
  <footer>Made with Material</footer>
</body></html>
"""


@pytest.fixture(scope="module")
def parsed():
    return DocumentParser().parse(_HTML)


def test_title_without_permalink(parsed):
    assert parsed.title == "Page"


def test_no_nav_or_footer_leakage(parsed):
    joined = " ".join(b.text for b in parsed.blocks)
    assert "Skip to content" not in joined
    assert "Made with Material" not in joined


def test_heading_path_tracks_sections(parsed):
    body = next(b for b in parsed.blocks if b.text.startswith("Body with"))
    assert body.heading_path == ("Page", "Section")


def test_prose_whitespace_normalized(parsed):
    body = next(b for b in parsed.blocks if b.text.startswith("Body"))
    assert body.text == "Body with messy whitespace."


def test_code_whitespace_preserved(parsed):
    code = next(b for b in parsed.blocks if b.type == "code")
    assert code.text == "x = 1\n\ny = 2"


def test_admonition_content_kept_title_dropped(parsed):
    texts = [b.text for b in parsed.blocks]
    assert "Useful hint." in texts
    assert "Tip" not in texts


def test_list_rendered_with_nesting(parsed):
    lst = next(b for b in parsed.blocks if b.type == "list")
    assert lst.text == "- one\n- two\n  - nested"


def test_every_block_well_formed(parsed):
    for b in parsed.blocks:
        assert isinstance(b, DocumentBlock)
        assert isinstance(b.heading_path, tuple)
        assert b.text.strip()


@pytest.mark.parametrize("name", ["query-params", "first-steps", "python-types"])
def test_real_page_parses_cleanly(name):
    html = (FIXTURES / f"{name}.html").read_text()
    doc = DocumentParser().parse(html)

    assert doc.title and "¶" not in doc.title
    assert len(doc.blocks) > 20

    joined = " ".join(b.text for b in doc.blocks)
    assert "Table of contents" not in joined
    assert "Skip to content" not in joined

    for b in doc.blocks:
        assert b.text.strip()
        assert b.heading_path[0] == doc.title  # h1 is the root of every path
        if b.type == "code":
            assert "<span" not in b.text and "<font" not in b.text


def test_known_heading_path_in_query_params():
    doc = DocumentParser().parse((FIXTURES / "query-params.html").read_text())
    paths = {b.heading_path for b in doc.blocks}
    assert ("Query Parameters", "Defaults") in paths
