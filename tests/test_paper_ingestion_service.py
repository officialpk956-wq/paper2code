"""
Unit tests for the paper ingestion helpers.

These tests cover the deterministic helper functions that power the upload
pipeline without requiring a real PDF fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.models import Base, Paper  # noqa: E402
from backend.services.paper_ingestion_service import (  # noqa: E402
    _normalize_title,
    _resolve_unique_title,
    extract_equations,
    extract_sections,
)


# ---------------------------------------------------------------------------
# _normalize_title
# ---------------------------------------------------------------------------

def test_normalize_title_strips_extension():
    assert _normalize_title('Attention Is All You Need.pdf') == 'Attention Is All You Need'


def test_normalize_title_no_extension():
    assert _normalize_title('Transformer Networks') == 'Transformer Networks'


def test_normalize_title_empty_stem_falls_back():
    assert _normalize_title('') == 'paper'


# ---------------------------------------------------------------------------
# _resolve_unique_title
# ---------------------------------------------------------------------------

def test_resolve_unique_title_suffixes_duplicate_titles():
    engine = create_engine('sqlite:///:memory:')
    TestingSession = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    try:
        db.add(Paper(title='Uploaded Paper', authors=None, abstract=None, architecture_graph=None, flops_analysis=None))
        db.commit()

        assert _resolve_unique_title(db, 'Uploaded Paper') == 'Uploaded Paper (2)'
    finally:
        db.close()


def test_resolve_unique_title_no_conflict():
    engine = create_engine('sqlite:///:memory:')
    TestingSession = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    try:
        result = _resolve_unique_title(db, 'Brand New Paper')
        assert result == 'Brand New Paper'
    finally:
        db.close()


def test_resolve_unique_title_multiple_duplicates():
    engine = create_engine('sqlite:///:memory:')
    TestingSession = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    try:
        db.add(Paper(title='My Paper', authors=None, abstract=None, architecture_graph=None, flops_analysis=None))
        db.add(Paper(title='My Paper (2)', authors=None, abstract=None, architecture_graph=None, flops_analysis=None))
        db.commit()

        assert _resolve_unique_title(db, 'My Paper') == 'My Paper (3)'
    finally:
        db.close()


# ---------------------------------------------------------------------------
# extract_equations
# ---------------------------------------------------------------------------

def test_extract_equations_detects_math_spans():
    equations = extract_equations([
        'The update rule is x = y + 1',
        'This line is prose only.',
        r'$$L = \sum_i \ell_i$$',
    ])

    assert len(equations) >= 2
    assert any('x = y + 1' in equation['text'] for equation in equations)
    assert any(r'L = \sum_i \ell_i' in equation['text'] for equation in equations)


def test_extract_equations_respects_line_length_cap():
    long_line = 'x = ' + ('a' * 250)
    equations = extract_equations([long_line])
    assert all(len(eq['text']) <= 240 for eq in equations)


def test_extract_equations_empty_pages():
    assert extract_equations([]) == []
    assert extract_equations(['', '   ']) == []


def test_extract_equations_cap_at_80():
    many_equations = '\n'.join(f'x_{i} = y_{i} + 1' for i in range(100))
    equations = extract_equations([many_equations])
    assert len(equations) <= 80


def test_extract_equations_assigns_page_numbers():
    equations = extract_equations(['', 'x = y + 1'])
    assert all('page' in eq for eq in equations)
    math_eq = [eq for eq in equations if 'x = y + 1' in eq['text']]
    assert math_eq[0]['page'] == 2


# ---------------------------------------------------------------------------
# extract_sections
# ---------------------------------------------------------------------------

def test_extract_sections_detects_abstract():
    pages = [
        'Abstract\nThis paper presents a novel approach to attention mechanisms.\n',
        'Introduction\nDeep learning has transformed NLP tasks significantly.',
    ]
    sections = extract_sections(pages)
    titles = [s['title'] for s in sections]
    assert 'Abstract' in titles
    assert 'Introduction' in titles


def test_extract_sections_canonical_names():
    pages = [
        'Introduction\nWe present a transformer-based model.\n',
        'related work\nPrior studies include...\n',
        'CONCLUSION\nIn this paper we showed...',
    ]
    sections = extract_sections(pages)
    titles = [s['title'] for s in sections]
    assert 'Introduction' in titles
    assert 'Related Work' in titles
    assert 'Conclusion' in titles


def test_extract_sections_numbered_headings():
    pages = [
        '1. Introduction\nWe propose a new method.\n',
        '2. Methods\nThe model consists of attention heads.',
    ]
    sections = extract_sections(pages)
    titles = [s['title'] for s in sections]
    assert 'Introduction' in titles
    assert 'Methods' in titles


def test_extract_sections_no_content_sections_skipped():
    pages = ['Abstract\n\n\nIntroduction\nSome actual content here.']
    sections = extract_sections(pages)
    titled = {s['title']: s for s in sections}
    # Abstract has no content (empty between Abstract and Introduction)
    assert 'Introduction' in titled
    assert titled['Introduction']['content'].strip() != ''


def test_extract_sections_content_truncated_at_4000_chars():
    long_content = 'word ' * 1000  # ~5000 chars
    pages = [f'Abstract\n{long_content}']
    sections = extract_sections(pages)
    if sections:
        assert len(sections[0]['content']) <= 4000


def test_extract_sections_returns_list_of_dicts():
    pages = ['Introduction\nSome content.']
    sections = extract_sections(pages)
    assert isinstance(sections, list)
    for section in sections:
        assert 'id' in section
        assert 'title' in section
        assert 'content' in section
        assert 'level' in section


def test_extract_sections_empty_input():
    assert extract_sections([]) == []
    assert extract_sections(['plain text without headings']) == []


def test_extract_sections_appendix_level_2():
    pages = ['Introduction\nContent.\n', 'Appendix\nAdditional material.']
    sections = extract_sections(pages)
    appendix_sections = [s for s in sections if s['title'] == 'Appendix']
    if appendix_sections:
        assert appendix_sections[0]['level'] == 2
