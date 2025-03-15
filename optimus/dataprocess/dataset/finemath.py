import pyarrow.parquet as pq

from pathlib import Path
from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(file) for file in Path(path).rglob("*.parquet")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    f = pq.ParquetFile(file_path, pre_buffer=True, buffer_size=8192)

    def _process_record(r: dict[str, Any]) -> dict[str, Any]:
        text = r["text"]
        metadata = {"id": r["url"]}
        return {"text": text, "metadata": metadata}

    for batch in f.iter_batches(batch_size=batch_size, columns=["text", "url"]):
        records = batch.to_pylist()
        processed = [_process_record(r) for r in records]
        yield processed

    f.close()
