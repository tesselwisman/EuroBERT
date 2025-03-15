import gzip
import json

from pathlib import Path
from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(file) for file in Path(path).rglob("*.jsonl.gz")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    with gzip.open(file_path, "rt") as f:
        batch = []
        for line in f:
            obj = json.loads(line)
            batch.append(
                {
                    "text": obj["text"],
                    "metadata": {
                        "id": obj["id"],
                        "industry_type": obj["industry_type"],
                    },
                }
            )

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
