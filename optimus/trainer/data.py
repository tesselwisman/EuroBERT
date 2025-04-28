import json
from types import MethodType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from streaming import Stream, StreamingDataLoader, StreamingDataset

from optimus.trainer.configuration.configs import Config


class Data:
    """Manages dataset loading, dataloader creation, and state management."""

    def __init__(self, config: Config, tokenizer):
        """
        Initialize the Data object with configurations and tokenizer. Load data mix and create datasets and dataloaders.
        Args:
            config (Config): Configuration object containing all the necessary configurations.
            tokenizer: Tokenizer object to be used for tokenizing the data.
        """
        self.data_config = config.data
        self.system_config = config.system
        self.main_process = config.is_main_process
        self.tokenizer = tokenizer
        self.hf_model = config.model.huggingface_id is not None

        assert (
            config.train.mask_probability
            + config.train.random_probability
            + config.train.original_probability
            == 1.0
        ), "The sum of masking probabilities must be equal to 1.0."
        self.num_canonical_nodes = (
            config.system.num_nodes
            if config.data.num_canonical_nodes <= 0
            else config.data.num_canonical_nodes
        )
        self.mlm_probability = config.train.mlm_probability
        self.mask_probability = config.train.mask_probability
        self.random_probability = (
            config.train.random_probability / (1 - self.mask_probability)
            if self.mask_probability < 1.0
            else 0.0
        )

        if config.data.num_canonical_nodes <= 0:
            config.update_config(num_canonical_nodes=self.num_canonical_nodes)

        # Load data mix for training & evaluation
        self.train_streams = self.__load_data_mix(
            rf"{self.data_config.data_mix_path}/train.json"
        )
        self.train_dataset = self.__create_dataset(self.train_streams)

        config.log_print("Train dataset created successfully:", len(self.train_dataset))

        # Create dataloaders
        self.train_dataloader = self.__create_dataloader(self.train_dataset)

        config.log_print(
            "Train dataloader created successfully:", len(self.train_dataloader)
        )
        try:
            self.eval_streams = self.__load_data_mix(
                rf"{self.data_config.data_mix_path}/eval.json"
            )
            self.eval_dataset = self.__create_dataset(self.eval_streams, eval=True) if self.eval_streams else None
            self.eval_dataloader = self.__create_dataloader(self.eval_dataset, self.eval_streams) if self.eval_dataset else None
            config.log_print("Eval dataloader created with length", len(self.eval_dataloader))
        except Exception:
            config.log_print("Evaluation data could not be loaded. If you expect data to be present, check your eval.json")
            self.eval_dataloader = None

        config.log_print(
            f"Masking probabilities: MLM={config.train.mlm_probability}, Mask={config.train.mask_probability}, Random={config.train.random_probability}, Original={config.train.original_probability}"
        )
        config.log_print(f"Number of canonical nodes: {self.num_canonical_nodes}")

    def __load_data_mix(self, path: str) -> list[Stream]:
        """
        Load data mix from the provided path.
        Args:
            path (str): Path to the data mix file.
        Returns:
            list[Stream]: List of Stream objects containing the data mix.
        """
        with open(path, "r") as file:
            streams_data = json.load(file)
            return [Stream(**item) for item in streams_data]

    def __create_dataset(
        self, streams: list[Stream], eval: bool = False
    ) -> StreamingDataset:
        """
        Create a dataset from the provided streams.
        Args:
            streams (list[Stream]): List of Stream objects containing the data
            eval (bool): Flag to determine if the dataset is for evaluation.
        Returns:
            Dataset: Dataset object containing the data.
        """
        return MaskingDataset(
            streams=streams,
            shuffle=False if eval else self.data_config.shuffle,
            shuffle_seed= 9176 if eval else self.data_config.seed,
            batch_size=self.data_config.batch_size,
            num_canonical_nodes=self.num_canonical_nodes,
            shuffle_block_size=int(max(4000000 // self.num_canonical_nodes, 1 << 18)),
            predownload=self.data_config.predownload * self.data_config.batch_size,
            mlm_probability=self.mlm_probability,
            mask_probability=self.mask_probability,
            random_probability=self.random_probability,
            tokenizer=self.tokenizer,
        )

    def __create_dataloader(self, dataset: StreamingDataset, eval: bool = False) -> StreamingDataLoader:
        """
        Create a dataloader from the provided streams.
        Args:
            streams (list[Stream]): List of Stream objects containing the data mix.
        Returns:
            DataLoader: DataLoader object containing the data mix.
        """
        dataloader = StreamingDataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            num_workers = self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor or None,
            collate_fn=self.to_torch_collate_var_len_fn
            if self.data_config.var_len
            else self.to_torch_collate_fn,
            pin_memory=self.data_config.pin_memory,
            drop_last=True,
            persistent_workers=False
        )

        dataloader._get_batch_size = MethodType(_get_batch_size, dataloader)
        return dataloader

    def to_torch_collate_fn(self, batch):
        """
        Collate function for the dataloader. This function is used to prepare the batch for training. All samples should be of the same length.
        Args:
            batch: Batch of data to be collated.
        Returns:
            dict[str, Any]: A dictionary containing the input_ids and labels for the batch.
        """
        batch = np.array(batch)
        inputs = torch.tensor(batch[:, 0], dtype=torch.long)
        labels = torch.tensor(batch[:, 1], dtype=torch.long)
        return {
            "input_ids" if self.hf_model else "x": inputs,
            "labels": labels,
        }

    def to_torch_collate_var_len_fn(self, batch):
        """
        Collate function for the dataloader. Prepares the batch for training with variable-length samples.
        Args:
            batch: List of tuples (input_seq, label_seq).
        Returns:
            dict[str, torch.Tensor]: A dictionary containing input_ids, labels, and cu_seq_lens.
        """
        # Unpack the sequences and labels
        input_seqs, label_seqs = zip(*batch)

        # Compute cumulative sequence lengths
        lengths = torch.tensor([len(seq) for seq in input_seqs], dtype=torch.int32)
        cu_seq_lens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(lengths, dim=0, dtype=torch.int32),
            ]
        )

        # Concatenate inputs and labels into single tensors
        inputs = torch.cat(
            [torch.tensor(seq, dtype=torch.long) for seq in input_seqs], dim=0
        )
        labels = torch.cat(
            [torch.tensor(seq, dtype=torch.long) for seq in label_seqs], dim=0
        )

        return {
            "input_ids" if self.hf_model else "x": inputs,
            "labels": labels,
            "cu_seq_lens": cu_seq_lens,
            "max_seqlen": torch.tensor(lengths.max().item(), dtype=torch.int32),
        }

    # ----------------------
    # Masking Dataset
    # ----------------------


class MaskingDataset(StreamingDataset):
    def __init__(
        self,
        mlm_probability: float,
        mask_probability: float,
        random_probability: float,
        tokenizer,
        *args,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        super(MaskingDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        inputs, labels = self.__masking_function(item["tokens"])
        return inputs, labels

    def __masking_function(self, item: Any) -> dict[str, Any]:
        """
        Prepare masked token inputs and labels for masked language modeling.

        This function is inspired by the Huggingface DataCollatorForLanguageModeling.
        The original function can be found at this URL: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py

        Args:
            inputs: Input data to be masked.

        Returns:
            inputs: A list containing the input_ids with masked tokens.
            labels: A list containing the original token for masked tokens, and -100 otherwise.
        """
        # Clone inputs to create labels
        inputs = np.copy(item)
        labels = np.copy(item)

        # We sample a few tokens in each sequence for MLM training.
        probability_matrix = np.full(labels.shape, self.mlm_probability)

        special_tokens_mask = np.array(
            self.tokenizer.get_special_tokens_mask(
                labels, already_has_special_tokens=True
            ),
            dtype=bool,
        )
        probability_matrix[special_tokens_mask] = 0.0

        masked_indices = np.random.rand(*probability_matrix.shape) < probability_matrix
        labels[~masked_indices] = -100  # We only compute loss on masked tokens.

        # mask_probability of the time, we replace masked input tokens with mask_token ([MASK])
        indices_replaced = (
            np.random.rand(*labels.shape) < self.mask_probability
        ) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # random_probability of the time, we replace masked input tokens with random word
        indices_random = (
            (np.random.rand(*labels.shape) < self.random_probability)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(0, len(self.tokenizer), size=labels.shape)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time we keep the masked input tokens unchanged
        return inputs, labels


# -------------------------
# Patch Streaming functions
# -------------------------


def _get_batch_size(self, batch: Any) -> int:
    """Get the number of samples in a batch.

    Args:
        _ (Any): _.

    Returns:
        int: Number of samples.
    """
    return self.batch_size


def patch_spanner():
    """Patches the Spanner class to use the new implementation of the SpannerPatch class."""
    from streaming.base import spanner

    spanner.Spanner.__init__ = SpannerPatch.__init__
    spanner.Spanner.__getitem__ = SpannerPatch.__getitem__


class SpannerPatch:
    """Patches the large memory allocation in the original Spanner initialization.

    This implementation was taken from: https://github.com/mosaicml/streaming/pull/773

    Below is the original docstring.

    Given a list of shards, construct a mapping of global index to shard and relative index.
    Args:
        shard_sizes (NDArray[np.int64]): Number of samples in each shard.
        span_size (int): Size of the divisions of the sample space. Defaults to ``1 << 10``.
    """

    def __init__(
        self, shard_sizes: NDArray[np.int64], span_size: int = 1 << 10
    ) -> None:
        self.shard_sizes = shard_sizes
        self.span_size = span_size
        self.num_samples = sum(shard_sizes)
        self.shard_bounds = np.concatenate(
            [np.zeros(1, np.int64), shard_sizes.cumsum()]
        )
        overflow = self.num_samples % span_size
        underflow = span_size - overflow if overflow else 0
        self.shard_sizes[-1] += underflow

        n_shards = len(shard_sizes)
        current_shard = 0
        current_position_in_shard = 0

        span_lowest_shards = []
        span_highest_shards = []

        while current_shard < n_shards:
            span_min_shard = current_shard
            span_max_shard = current_shard

            remaining_span_size = span_size
            while remaining_span_size > 0 and current_shard < n_shards:
                available_in_current_shard = (
                    shard_sizes[current_shard] - current_position_in_shard
                )

                if remaining_span_size >= available_in_current_shard:
                    remaining_span_size -= available_in_current_shard
                    current_shard += 1
                    current_position_in_shard = 0
                else:
                    current_position_in_shard += remaining_span_size
                    remaining_span_size = 0

                if current_shard < n_shards:
                    span_max_shard = current_shard

            span_lowest_shards.append(span_min_shard)
            span_highest_shards.append(span_max_shard)

        self.spans = []
        for low, high in zip(span_lowest_shards, span_highest_shards):
            shards = np.arange(low, high + 1)
            self.spans.append(shards)
        self.shard_sizes[-1] -= underflow

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Map global sample index to shard and relative sample index.
        Args:
            index (int): Global sample index.
        Returns:
            Tuple[int, int]: Shard and relative sample index.
        """
        if not (0 <= index < self.num_samples):
            raise IndexError(
                f"Invalid sample index `{index}`: 0 <= {index} < {self.num_samples}"
            )
        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_bounds[shard]
            shard_stop = self.shard_bounds[shard + 1]
            if shard_start <= index < shard_stop:
                return shard, int(index - shard_start.item())  # pyright: ignore
        raise RuntimeError("Internal error: shards were indexed incorrectly")

