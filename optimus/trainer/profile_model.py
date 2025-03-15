import fire
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from optimus.trainer.configuration.configs import Config
from optimus.trainer.model.load import load_model


def main(**kwargs):
    # Load configurations
    config = Config(**kwargs)

    assert not config.use_fsdp and not config.use_ddp, "FSDP and DDP not supported."

    # Load/set model and get tokenizer.
    model = load_model(config)

    batch_size = config.data.batch_size or 1
    seq_length = config.model.block_size or 512
    vocab_size = config.model.vocab_size or 1000
    num_samples = 100

    dummy_data = [
        torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long).cuda()
        for _ in range(num_samples)
    ]

    model.eval()

    output_dir = config.train.output_dir or "output"

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace(f"{output_dir}/trace_{p.step_num}.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # Repeat 0 for infinite profiling
        schedule=schedule(wait=5, warmup=2, active=3, repeat=0),
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        with torch.no_grad():
            for input_ids in dummy_data:
                _ = model(input_ids)
                prof.step()


if __name__ == "__main__":
    fire.Fire(main)
