import pandas as pd

from pathlib import Path
from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(file) for file in Path(path).rglob("*.csv")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    lang_pair = file_path.split("/")[-2]
    src_lang, tgt_lang = lang_pair.split("-")

    def _process_record(r: dict[str, Any]) -> dict[str, Any]:
        src, tgt = r[src_lang], r[tgt_lang]
        r_pr = {"text": f"{src}<|parallel_sep|>{tgt}", "metadata": ""}
        return r_pr

    for chunk in pd.read_csv(
        file_path, usecols=[src_lang, tgt_lang], chunksize=batch_size
    ):
        records = chunk.to_dict(orient="records")
        processed = []
        for r in records:
            processed.append(_process_record(r))
        yield processed
