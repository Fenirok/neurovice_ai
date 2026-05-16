from pathlib import Path
from functools import lru_cache
import joblib


def _load_artifact(path: Path):
    """
    Internal artifact loader with validation.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact not found: {path}"
        )

    return joblib.load(path)


@lru_cache(maxsize=8)
def load_model(path: Path):
    """
    Cached model loader.
    Prevents repeated disk I/O.
    """

    return _load_artifact(path)


@lru_cache(maxsize=8)
def load_scaler(path: Path):
    """
    Cached scaler loader.
    Prevents repeated disk I/O.
    """

    return _load_artifact(path)