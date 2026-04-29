import numpy as np
import pytest

from code.embeddings.cache import EmbeddingCache, embeddings_for_ids


def test_cache_round_trip(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "emb.npz"))
    data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0]), "c": np.array([5.0, 6.0])}
    cache.put(data)

    found, missing = cache.get(["a", "b", "c"])

    assert missing == []
    for key in data:
        np.testing.assert_array_equal(found[key], data[key])


def test_cache_partial_miss(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "emb.npz"))
    cache.put({"a": np.array([1.0]), "b": np.array([2.0])})

    found, missing = cache.get(["a", "b", "c"])

    assert set(found.keys()) == {"a", "b"}
    assert missing == ["c"]


def test_cache_persists_to_disk(tmp_path):
    path = str(tmp_path / "emb.npz")
    cache1 = EmbeddingCache(path)
    data = {"x": np.array([7.0, 8.0])}
    cache1.put(data)

    cache2 = EmbeddingCache(path)
    found, missing = cache2.get(["x"])

    assert missing == []
    np.testing.assert_array_equal(found["x"], data["x"])


def test_embeddings_for_ids_calls_encoder_only_for_missing(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "emb.npz"))
    cache.put({"a": np.array([1.0, 2.0])})

    calls = []
    def mock_encoder(ids):
        calls.append(ids)
        return np.ones((len(ids), 2))

    result = embeddings_for_ids(["a", "b", "c"], cache, mock_encoder)

    assert len(calls) == 1
    assert set(calls[0]) == {"b", "c"}
    assert "a" not in calls[0]
    assert set(result.keys()) == {"a", "b", "c"}


def test_embeddings_for_ids_puts_new_entries_in_cache(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "emb.npz"))

    embeddings_for_ids(["b", "c"], cache, lambda ids: np.ones((len(ids), 4)))

    found, missing = cache.get(["b", "c"])
    assert missing == []
    assert "b" in found and "c" in found
