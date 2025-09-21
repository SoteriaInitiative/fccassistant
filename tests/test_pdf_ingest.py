import re

import model.pdf_ingest as pi


def test_detect_source_ofac_vs_fatf():
    ofac_text = "This guidance from the Office of Foreign Assets Control (OFAC) addresses FAQs."
    fatf_text = "The Financial Action Task Force (FATF) Recommendations outline global standards."
    assert pi._detect_source(ofac_text) == "ofac"
    assert pi._detect_source(fatf_text) == "fatf"


def test_clean_pdf_text_dehyphenate_and_spaces():
    raw = "Inter-\nnational  sanctions\n\n\nNext  line\rCarriage\rreturn"
    out = pi._clean_pdf_text(raw)
    # dehyphenation and newline normalization
    assert "International" in out
    # multiple blank lines collapsed
    assert "\n\n\n" not in out


def test_ofac_header_chunking_minimal():
    text = (
        "1. What is a sanction?\nIt is a measure.\n\n"
        "2. How are sanctions enforced?\nBy authorities.\n"
    )
    chunks = pi._chunk_ofac_faq(text)
    # Expect two question-anchored chunks
    assert len(chunks) == 2
    assert chunks[0].startswith("1. What is a sanction?")
    assert chunks[1].startswith("2. How are sanctions enforced?")


def test_fallback_chunks_basic_windowing():
    body = "abcdefghijklmnopqrstuvwxyz" * 10
    ch = pi._fallback_chunks(body, size=50, overlap=10)
    # Windowed with overlap
    assert len(ch) >= 5
    assert all(len(c) <= 50 for c in ch)


def test_chunk_text_uses_specialized_then_fallback():
    # OFAC-like cue should use ofac header chunking
    ofac_doc = "1. A question?\nAnswer para 1.\n\nAnswer para 2.\n\n2. Another question?\nMore text."
    ofac_chunks = pi.chunk_text(ofac_doc, size=200, overlap=20, source="ofac")
    assert len(ofac_chunks) >= 2
    assert ofac_chunks[0].split("\n", 1)[0].startswith("1.")

    # Unknown content should fall back
    unknown = "This is plain text without recognizable headers." * 10
    fb = pi.chunk_text(unknown, size=120, overlap=30, source="unknown")
    assert len(fb) >= 3
    assert all(len(c) > 0 for c in fb)

