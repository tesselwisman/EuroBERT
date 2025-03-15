EuroBERT Data Processing Documentation
======================================

This documentation covers the data processing tools provided by the EuroBERT training library (optimus). These tools enable efficient preprocessing for large-scale language model training, including tokenizing, packing, subsampling, and inspecting datasets.

Table of Contents
-----------------

*   [1\. Tokenizing a Dataset](#tokenizing-a-dataset)
*   [2\. Packing a Dataset (Optional)](#packing-a-dataset-optional)
*   [3\. Subsampling a Dataset (Optional)](#subsampling-a-dataset-optional)
*   [4\. Inspecting a Dataset (Optional)](#inspecting-a-dataset-optional)

1\. Tokenizing a Dataset
------------------------

The `tokenize_dataset.py` script tokenizes a dataset using a specified tokenizer and saves the output in an optimized format. The tokenized data can be split into shards based on a size limit and processed in parallel using multiple workers.

### Usage

    python tokenize_dataset.py --input_dir <path> --tokenizer <path_or_name> --dataset <name> [--output_dir <path>] [--size_limit <value>] [--num_workers <num>] [--head <num>] [--read_files_kwargs <json>] [--timeout <seconds>] [--tiktoken]

### Parameters

*   `input_dir` (_str_): Path to the directory containing the input dataset.
*   `tokenizer` (_str_): Name or path of the tokenizer to be used.
*   `dataset` (_str_): Name of the dataset to process.
*   `output_dir` (_str_, optional): Directory to save the tokenized dataset. Defaults to `./output`.
*   `size_limit` (_int | str_, optional): Maximum shard size before creating a new shard. Supports human-readable formats (e.g., `100kb`, `1mb`). Defaults to `64MB` (`1 << 26`).
*   `num_workers` (_int | str_, optional): Number of worker processes. Can be set to `max` to use all available CPUs. Defaults to `1`.
*   `head` (_int_, optional): Number of batches to process. If not specified, processes the entire dataset.
*   `read_files_kwargs` (_dict[str, Any]_, optional): Additional parameters for dataset file reading. Defaults to `None`.
*   `timeout` (_int_, optional): Maximum time (in seconds) before termination. Defaults to `None` (no timeout).
*   `tiktoken` (_bool_, optional): Whether to use `tiktoken` for faster tokenization. Defaults to `False`.

### Example

    python tokenize_dataset.py --input_dir ./codebagel --tokenizer "meta-llama/Llama-3.1-8B-Instruct" --dataset codeBagel --output_dir ./output --size_limit 100mb --num_workers 4

This command tokenizes the `codeBagel` dataset stored in `./codebagel` using the `meta-llama/Llama-3.1-8B-Instruct` tokenizer, saves the output in `./output`, splits shards at 100MB, and runs with 4 workers.

#### Fastest Tokenization with `tiktoken`

For maximum efficiency (up to 2x speedup), use `tiktoken` with a compatible tokenizer and `tokenizer.model` [file](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/original/tokenizer.model):

    python tokenize_dataset.py --input_dir ./codebagel --tokenizer ./llama_tokenizer.model --dataset codeBagel --output_dir ./tokenized --size_limit 100mb --num_workers 4 --tiktoken

---

2\. Packing a Dataset (Optional)
--------------------------------

The `pack_dataset.py` script packs a dataset into blocks (sentences) of a fixed size or uniformly distributed sentence length.

### Usage

    python pack_dataset.py --local_dir <path> --output_dir <path> [--block_size <int>] [--val_size <int>] [--size_limit <int|str>] [--head <int>] [--num_workers <int>] [--random_size]

### Parameters

*   `local_dir` (_str_): Local directory path containing the tokenized dataset.
*   `output_dir` (_str_): Directory to save the packed dataset.
*   `block_size` (_int_, optional): Block size for packing. Defaults to `None`.
*   `val_size` (_int_, optional): Validation set size. Defaults to `None`.
*   `size_limit` (_int | str_, optional): Size limit for the output files. Defaults to `"64MB"`.
*   `head` (_int_, optional): Number of records to process. Defaults to `None`.
*   `num_workers` (_int_, optional): Number of workers. Defaults to `1`.
*   `random_size` (_bool_, optional): If `True`, selects a random block size between `12` and `block_size`, resulting in a uniformly distributed sentence length. Defaults to `False`.

### Example

    python pack_dataset.py --local_dir './tokenized' --output_dir './output_pack' --block_size 2048 --random_size

---

3\. Subsampling a Dataset (Optional)
-------------------------

The `subsample_dataset.py` script processes a dataset to split it into subdirectories and merges the indexes. This is particularly useful for optimizing dataset loading, especially in environments like MosaicML, to avoid GPU synchronization timeout errors.

### Usage

    python subsample_dataset.py --dataset_path <path> --num_shards <int>

### Parameters

*   `dataset_path` (_str_): The base path where the datasets are located.
*   `num_shards` (_int_): The number of shards to split the dataset into.

### Example

    python subsample_dataset.py --dataset_path "./tokenized" --num_shards 2

---

4\. Inspecting a Dataset (Optional)
-----------------------------------

The `inspect_dataset.py` script inspects a processed dataset by printing a few samples.

### Usage

    python inspect_dataset.py --local_dir <path> --tokenizer <path_or_name> [--num_samples <int>]

### Parameters

*   `local_dir` (_str_): Dataset directory path.
*   `tokenizer_name` (_str_): Tokenizer name or path.
*   `num_samples` (_int_, optional): Number of samples to print. Defaults to `5`.

### Example

    python inspect_dataset.py --local_dir './output' --tokenizer "meta-llama/Llama-3.1-8B-Instruct" --num_samples 5