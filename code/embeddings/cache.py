from typing import Callable

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str) -> None:
        self._path = path
        self._data: dict[str, np.ndarray] = {}
        try:
            loaded = np.load(path, allow_pickle=False)
            self._data = {k: loaded[k] for k in loaded.files}
        except FileNotFoundError:
            pass

    def get(self, keys: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        found = {k: self._data[k] for k in keys if k in self._data}
        missing = [k for k in keys if k not in self._data]
        return found, missing

    def put(self, embeddings: dict[str, np.ndarray]) -> None:
        self._data.update(embeddings)
        np.savez(self._path, **self._data)


def embeddings_for_ids(
    ids: list[str],
    cache: EmbeddingCache,
    encoder_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, np.ndarray]:
    found, missing = cache.get(ids)
    if missing:
        new_embs = encoder_fn(missing)
        new_dict = {k: new_embs[i] for i, k in enumerate(missing)}
        cache.put(new_dict)
        found.update(new_dict)
    return found
