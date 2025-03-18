import torch
import torch.nn as nn

from optimus.trainer.model import model

bert_config = {
    "280m": {
        "vocab_size": 30522,
        "embedding_size": 768,
        "num_head": 12,
        "num_kv_head": 12,
        "num_layer": 12,
        "block_size": 512,
        "dropout": 0.1,
        "mlp_hidden_dim": 3072,
        "bias": False,
        "attn_impl": "torch",
    },
    "3b": {
        "vocab_size": 30522,
        "embedding_size": 2560,
        "num_head": 32,
        "num_kv_head": 8,
        "num_layer": 32,
        "block_size": 512,
        "dropout": 0.1,
        "mlp_hidden_dim": 10240,
        "bias": False,
        "attn_impl": "torch",
    },
}


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = model.CustomEmbedding(
            config["vocab_size"], config["embedding_size"]
        )
        self.positional_embedding = model.CustomEmbedding(
            config["block_size"], config["embedding_size"]
        )

    def forward(self, input_ids):
        input_embedding = self.embedding(input_ids)
        position_embedding = self.positional_embedding(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        embedding = input_embedding + position_embedding
        return embedding


class Bert(model.TransformerEncoder):
    def __init__(self, config=bert_config["280m"]):
        super().__init__(
            embedding=BertEmbedding(config),
            blocks=[
                model.Block(
                    attention=get_attn_cls(config["attn_impl"])(
                        embed_dim=config["embedding_size"],
                        head_dim=config["embedding_size"] // config["num_head"],
                        num_heads=config["num_head"],
                        num_kv_heads=config["num_kv_head"],
                        dropout=config["dropout"],
                        block_size=config["block_size"],
                        rope=None,
                        bias=config["bias"],
                    ),
                    mlp=model.GeluMLP(
                        embed_dim=config["embedding_size"],
                        hidden_dim=config["mlp_hidden_dim"],
                        bias=config["bias"],
                    ),
                    attn_norm=nn.LayerNorm(
                        config["embedding_size"], bias=config["bias"]
                    ),
                    mlp_norm=nn.LayerNorm(
                        config["embedding_size"], bias=config["bias"]
                    ),
                    dropout=config["dropout"],
                )
                for _ in range(config["num_layer"])
            ],
            final_layernorm=nn.LayerNorm(config["embedding_size"], bias=config["bias"]),
            lm_head=nn.Linear(
                config["embedding_size"], config["vocab_size"], bias=config["bias"]
            ),
        )
        self.name_or_path = "Bert"

    @property
    def device(self):
        return next(self.parameters()).device


def get_attn_cls(attn: str):
    if attn == "flash":
        return model.FlashSelfAttention
    elif attn == "torch":
        return model.TorchSelfAttention
    else:
        raise ValueError(f"Invalid attention type: {attn}")
