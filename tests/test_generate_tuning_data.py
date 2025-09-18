import json

import model.generate_tuning_data as gtd


def test_bucket_and_prefix_parsing():
    b, p = gtd._bucket_and_prefix("gs://my-bucket/foo/bar")
    assert b == "my-bucket"
    assert p == "foo/bar"
    b2, p2 = gtd._bucket_and_prefix("plain-bucket")
    assert b2 == "plain-bucket"
    assert p2 == ""


def test_naive_summary_truncates_and_keeps_sentences():
    text = "This is sentence one. This is sentence two. This is sentence three."
    s = gtd.naive_summary(text, max_chars=80)
    # Should include at least two sentences and be within limit
    assert "sentence one." in s
    assert "sentence two." in s
    assert len(s) <= 80


def test_normalize_row_and_make_row_shape():
    row = gtd.make_row("Question?", "Answer.")
    assert isinstance(row, dict)
    assert "contents" in row and len(row["contents"]) == 2
    user, model = row["contents"]
    assert user["role"] == "user"
    assert model["role"] == "model"
    assert user["parts"][0]["text"].endswith("Question?")
    assert model["parts"][0]["text"].endswith("Answer.")


def test_normalize_row_rejects_bad_shapes():
    bad1 = {"contents": []}
    bad2 = {"contents": [{"role": "user", "parts": []}, {"role": "model", "parts": [{"not_text": 1}]}]}
    try:
        gtd.normalize_row(bad1)
        assert False, "expected ValueError for empty contents"
    except ValueError:
        pass
    try:
        gtd.normalize_row(bad2)
        assert False, "expected ValueError for bad parts"
    except ValueError:
        pass

