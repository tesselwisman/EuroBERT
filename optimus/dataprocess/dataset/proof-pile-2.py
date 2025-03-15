import io
import json
from pathlib import Path
import zstandard as zstd

from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(file) for file in Path(path).rglob("*.jsonl.zst")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    DCTX = zstd.ZstdDecompressor()
    batch = []

    with (
        zstd.open(file_path, mode="rb", dctx=DCTX) as zfh,
        io.TextIOWrapper(zfh) as iofh,
    ):
        for line in iofh:
            obj = json.loads(line)
            batch.append({"text": obj["text"], "metadata": {}})
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
