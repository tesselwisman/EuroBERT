from dataclasses import dataclass


# System configuration (will be updated by the trainer, no need to change)
@dataclass
class SystemConfig:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    gpu_per_node: int = 1
    num_nodes: int = 1
    verbose: bool = True
