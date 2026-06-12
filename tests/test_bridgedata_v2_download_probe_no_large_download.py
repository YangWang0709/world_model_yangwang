from data.bridgedata_v2_download_probe import classify_download_candidate


def _candidate(url, content_length, size_known=True):
    return {
        "name": "tiny sample",
        "url": url,
        "link_text": "tiny sample",
        "probe": {
            "content_length_bytes": content_length,
            "size_known": size_known,
        },
    }


def test_fake_candidate_over_1gb_is_rejected():
    result = classify_download_candidate(_candidate("https://rail-berkeley.github.io/bridgedata/tiny.zip", 1073741825), 1073741824)
    assert result["safe_under_limit"] is False
    assert result["selected_safe"] is False
    assert result["unsafe_reason"] == "larger than max_download_bytes"


def test_fake_candidate_unknown_size_is_rejected():
    result = classify_download_candidate(_candidate("https://rail-berkeley.github.io/bridgedata/tiny.zip", None, False), 1073741824)
    assert result["size_known"] is False
    assert result["selected_safe"] is False
    assert result["unsafe_reason"] == "size unknown"


def test_fake_candidate_under_1gb_can_be_allowed_without_downloading():
    result = classify_download_candidate(_candidate("https://rail-berkeley.github.io/bridgedata/tiny_sample.zip", 1234), 1073741824)
    assert result["safe_under_limit"] is True
    assert result["selected_safe"] is True
    assert result["content_length_bytes"] == 1234
